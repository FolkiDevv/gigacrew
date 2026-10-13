"""Поведенческие сценарии (eval): прогон агента на живых моделях и метрики.

Общий модуль live-набора ``test_behavior_live.py``, проб шлюза ``test_api_probe_live.py``
и их офлайн-тестов ``test_behavior.py``:

* учебный проект ``shop`` (:func:`make_project`), на котором идут сценарии;
* «сценарный пользователь» (:class:`ScriptedUser`): на одобрение плана отвечает по
  сценарию, на прочие вопросы — первым вариантом, подтверждает правки файлов и только
  безопасные команды (:func:`is_safe_command`);
* запись обращений к модели (:class:`RecordingProvider`): размер запросов, кеш, вызовы;
* прогон сценария и метрики (:func:`run_scenario`), инварианты harness'а
  (:func:`invariant_errors`), отчёт (:class:`EvalSession`, :func:`summary_table`),
  выжимка для чата (:func:`digest_report`) и сравнение отчётов (:func:`compare_reports`);
* разбор ответов шлюза для проб (:func:`reasoning_texts`, :func:`with_reasoning`,
  :func:`describe_probe`).

Метрики берутся только из того, что агент и так отдаёт наружу (история, события, файлы
проекта), поэтому набор работает и на коде до изменений промпта — первый прогон даёт
baseline. Поведение модели случайно: сценарий повторяется, в отчёте — доли прогонов.

Просмотр, выжимка для чата (до ~100 строк) и сравнение сохранённых отчётов::

    uv run python tests/behavior.py show .gigacrew/evals/eval-<время>.json
    uv run python tests/behavior.py digest .gigacrew/evals/eval-<время>.json
    uv run python tests/behavior.py compare <было>.json <стало>.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fakes import RecordingEvents

from gigacrew.agent.context_window import CHARS_PER_TOKEN
from gigacrew.agent.loop import PLAN_BLOCKED_NOTE, REJECTED_NOTE, SELF_CHECK_NOTE, Agent
from gigacrew.agent.prompts import SUBAGENT_PROMPT
from gigacrew.config import Config
from gigacrew.llm.base import LLMProvider
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.project.index import INDEX_ERRORS, ProjectIndex
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import build_default_registry
from gigacrew.tools.plan import APPROVE_EDITS, PLAN_QUESTION
from gigacrew.tools.questions import Answer, Question

REPO_ROOT = Path(__file__).resolve().parent.parent

EVAL_MODELS_ENV = "GIGACREW_EVAL_MODELS"
EVAL_RUNS_ENV = "GIGACREW_EVAL_RUNS"
EVAL_LABEL_ENV = "GIGACREW_EVAL_LABEL"
EVAL_DIR_ENV = "GIGACREW_EVAL_DIR"
EVAL_TIMEOUT_ENV = "GIGACREW_EVAL_TIMEOUT"
DEFAULT_RUNS = 3
DEFAULT_TIMEOUT = 1800  # с на один прогон сценария

# Уведомления, которыми harness сообщает, что модель записала план, но не отправила его
# (до этапа 3 — напоминание модели, теперь — показ плана самим harness'ом): счёт ``rem``
# сравним между отчётами. Меняется текст — обновить здесь.
HARNESS_PLAN_NOTICES = ("не отправлен на одобрение",)
# Первая строка вопроса самопроверки: так её служебное сообщение видно в истории.
SELF_CHECK_MARK = SELF_CHECK_NOTE.splitlines()[0]
# Уведомление о паузе из-за лимита запросов шлюза (429).
RATE_LIMIT_MARK = "(429)"
# Инструменты исследования кода: их вызовы основным агентом — «работа вместо суб-агента».
RESEARCH_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "find_files",
        "search_content",
        "find_symbol",
        "find_references",
        "goto_definition",
        "call_hierarchy",
        "file_outline",
        "repo_map",
    }
)
MAX_ANSWER_CHARS = 2_000
# Начало системного промпта суб-агента: так запись запроса отличает его от основного.
_SUBAGENT_MARK = SUBAGENT_PROMPT[:60]


def eval_models(env: Mapping[str, str]) -> list[str]:
    """Модели eval: ``GIGACREW_EVAL_MODELS`` через запятую, иначе ``GIGACREW_TEST_MODEL``."""
    raw = env.get(EVAL_MODELS_ENV) or env.get("GIGACREW_TEST_MODEL") or ""
    return list(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Целое из окружения (не меньше 1); не задано — ``default``."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}: ожидается целое число, получено {raw!r}") from None
    return max(value, 1)


def report_dir(env: Mapping[str, str]) -> Path:
    """Папка отчётов: ``GIGACREW_EVAL_DIR``, по умолчанию ``.gigacrew/evals`` репозитория."""
    raw = (env.get(EVAL_DIR_ENV) or "").strip()
    return Path(raw).expanduser() if raw else REPO_ROOT / ".gigacrew" / "evals"


# --------------------------------------------------------------------------- #
# Учебный проект
# --------------------------------------------------------------------------- #
FIXTURE_FILES: dict[str, str] = {
    ".gitignore": "__pycache__/\n.pytest_cache/\n.gigacrew/\ndata/\n",
    "README.md": """\
# shop

Учебный интернет-магазин: корзина, заказы, оплата и хранилище заказов в JSON-файле.

- `shop/api.py` — HTTP-слой: маршруты и коды ответов;
- `shop/cart.py`, `shop/orders.py`, `shop/payments.py` — бизнес-логика;
- `shop/storage.py` — хранилище заказов;
- `shop/config.py` — настройки.

Тесты: `python -m pytest -q`.
""",
    "pyproject.toml": """\
[project]
name = "shop"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
""",
    "shop/__init__.py": '"""Учебный интернет-магазин."""\n',
    "shop/config.py": '''\
"""Настройки магазина."""

from dataclasses import dataclass
from pathlib import Path

MAX_CART_ITEMS = 50  # предел товаров в одной корзине
MAX_DISCOUNT_PERCENT = 30  # предел скидки, %


@dataclass(frozen=True)
class Settings:
    storage_path: Path = Path("data/orders.json")
    currency: str = "RUB"
    max_cart_items: int = MAX_CART_ITEMS
    max_discount_percent: int = MAX_DISCOUNT_PERCENT


def load_settings(env: dict[str, str]) -> Settings:
    """Настройки из переменных окружения (SHOP_STORAGE, SHOP_CURRENCY)."""
    return Settings(
        storage_path=Path(env.get("SHOP_STORAGE", "data/orders.json")),
        currency=env.get("SHOP_CURRENCY", "RUB"),
    )
''',
    "shop/errors.py": '''\
"""Исключения магазина. HTTP-слой превращает их в ответы (shop/api.py)."""


class ShopError(Exception):
    """Базовая ошибка магазина."""


class ValidationError(ShopError):
    """Некорректные входные данные."""


class NotFoundError(ShopError):
    """Товар или заказ не найден."""


class PaymentError(ShopError):
    """Платёж отклонён."""


class StorageError(ShopError):
    """Хранилище недоступно или повреждено."""
''',
    "shop/money.py": '''\
"""Деньги: суммы хранятся в копейках (int)."""

from shop.errors import ValidationError


def parse_price(text: str) -> int:
    """Цена из строки: '12.50' -> 1250. Пустая строка и мусор - ValidationError."""
    text = text.strip().replace(",", ".")
    if not text:
        raise ValidationError("пустая цена")
    rubles, _, kopecks = text.partition(".")
    try:
        value = int(rubles) * 100 + int((kopecks or "0").ljust(2, "0")[:2])
    except ValueError as e:
        raise ValidationError(f"некорректная цена: {text!r}") from e
    if value < 0:
        raise ValidationError("цена не может быть отрицательной")
    return value


def format_price(kopecks: int, currency: str = "RUB") -> str:
    """1250 -> '12.50 RUB'."""
    return f"{kopecks // 100}.{kopecks % 100:02d} {currency}"
''',
    "shop/models.py": '''\
"""Модели данных магазина."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    sku: str
    title: str
    price: int  # копейки


@dataclass
class CartItem:
    product: Product
    quantity: int = 1

    @property
    def total(self) -> int:
        return self.product.price * self.quantity


@dataclass
class Order:
    id: str
    items: list[CartItem] = field(default_factory=list)
    total: int = 0
    status: str = "new"  # new -> paid | failed
''',
    "shop/cart.py": '''\
"""Корзина покупателя."""

from shop.config import MAX_CART_ITEMS
from shop.errors import NotFoundError, ValidationError
from shop.models import CartItem, Product


class Cart:
    def __init__(self, max_items: int = MAX_CART_ITEMS):
        self.max_items = max_items
        self._items: dict[str, CartItem] = {}

    def add(self, product: Product, quantity: int = 1) -> None:
        """Добавляет товар; товаров в корзине не больше max_items."""
        if quantity <= 0:
            raise ValidationError("количество должно быть положительным")
        if self.count() + quantity > self.max_items:
            raise ValidationError(f"в корзине может быть не больше {self.max_items} товаров")
        item = self._items.get(product.sku)
        if item is None:
            self._items[product.sku] = CartItem(product, quantity)
        else:
            item.quantity += quantity

    def remove(self, sku: str) -> None:
        if sku not in self._items:
            raise NotFoundError(f"товара {sku} нет в корзине")
        del self._items[sku]

    def items(self) -> list[CartItem]:
        return list(self._items.values())

    def count(self) -> int:
        return sum(item.quantity for item in self._items.values())

    def total(self) -> int:
        return sum(item.total for item in self._items.values())
''',
    "shop/payments.py": '''\
"""Платёжный шлюз (заглушка внешнего сервиса)."""

from shop.errors import PaymentError

DECLINED_CARDS = {"4000000000000002"}


class PaymentGateway:
    def __init__(self) -> None:
        self.charges: list[tuple[str, int]] = []

    def charge(self, card: str, amount: int) -> str:
        """Списывает amount копеек с карты; возвращает id платежа."""
        if amount <= 0:
            raise PaymentError("сумма платежа должна быть положительной")
        if card in DECLINED_CARDS:
            raise PaymentError("платёж отклонён банком")
        self.charges.append((card, amount))
        return f"pay-{len(self.charges)}"
''',
    "shop/storage.py": '''\
"""Хранилище заказов в JSON-файле."""

import json
from pathlib import Path

from shop.errors import NotFoundError, StorageError
from shop.models import Order


class JsonStorage:
    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        if not self.path.exists():
            return {"orders": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise StorageError(f"не удалось прочитать {self.path}: {e}") from e

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            raise StorageError(f"не удалось записать {self.path}: {e}") from e

    def save_order(self, order: Order) -> None:
        data = self._read()
        data["orders"][order.id] = {"total": order.total, "status": order.status}
        self._write(data)

    def order_status(self, order_id: str) -> str:
        order = self._read()["orders"].get(order_id)
        if order is None:
            raise NotFoundError(f"заказ {order_id} не найден")
        return order["status"]
''',
    "shop/orders.py": '''\
"""Оформление заказов."""

import uuid

from shop.cart import Cart
from shop.errors import PaymentError, ValidationError
from shop.models import Order
from shop.payments import PaymentGateway
from shop.storage import JsonStorage


class OrderService:
    def __init__(self, storage: JsonStorage, payments: PaymentGateway):
        self.storage = storage
        self.payments = payments

    def checkout(self, cart: Cart, card: str) -> Order:
        """Оплачивает корзину и сохраняет заказ (и неудачный тоже)."""
        if not cart.items():
            raise ValidationError("корзина пуста")
        order = Order(id=uuid.uuid4().hex[:8], items=cart.items(), total=cart.total())
        try:
            self.payments.charge(card, order.total)
        except PaymentError:
            order.status = "failed"
            self.storage.save_order(order)
            raise
        order.status = "paid"
        self.storage.save_order(order)
        return order
''',
    "shop/api.py": '''\
"""HTTP-слой: маршруты и превращение ошибок магазина в ответы."""

from collections.abc import Callable

from shop.cart import Cart
from shop.errors import NotFoundError, PaymentError, ShopError, StorageError, ValidationError
from shop.models import Product
from shop.money import parse_price
from shop.orders import OrderService

STATUS_BY_ERROR: dict[type[ShopError], int] = {
    ValidationError: 400,
    PaymentError: 402,
    NotFoundError: 404,
    StorageError: 503,
}

Handler = Callable[[dict], dict]


def error_response(error: ShopError) -> tuple[int, dict]:
    """Код и тело ответа для ошибки магазина; неизвестная ошибка - 500."""
    for error_type, status in STATUS_BY_ERROR.items():
        if isinstance(error, error_type):
            return status, {"error": str(error)}
    return 500, {"error": "внутренняя ошибка"}


class App:
    def __init__(self, orders: OrderService):
        self.orders = orders
        self.cart = Cart()
        self.routes: dict[str, Handler] = {
            "cart.add": self.add_to_cart,
            "order.checkout": self.checkout,
            "order.status": self.order_status,
        }

    def handle(self, route: str, payload: dict) -> tuple[int, dict]:
        handler = self.routes.get(route)
        if handler is None:
            return 404, {"error": f"нет маршрута {route}"}
        try:
            return 200, handler(payload)
        except ShopError as e:
            return error_response(e)

    def add_to_cart(self, payload: dict) -> dict:
        product = Product(payload["sku"], payload.get("title", ""), parse_price(payload["price"]))
        self.cart.add(product, int(payload.get("quantity", 1)))
        return {"count": self.cart.count(), "total": self.cart.total()}

    def checkout(self, payload: dict) -> dict:
        order = self.orders.checkout(self.cart, payload["card"])
        self.cart = Cart()
        return {"order": order.id, "total": order.total}

    def order_status(self, payload: dict) -> dict:
        return {"status": self.orders.storage.order_status(payload["order"])}
''',
    "tests/test_money.py": """\
import pytest

from shop.errors import ValidationError
from shop.money import format_price, parse_price


def test_parse_price():
    assert parse_price("12.50") == 1250
    assert parse_price("7") == 700
    assert parse_price("3,5") == 350


def test_parse_price_rejects_garbage():
    with pytest.raises(ValidationError):
        parse_price("")
    with pytest.raises(ValidationError):
        parse_price("abc")


def test_format_price():
    assert format_price(1250) == "12.50 RUB"
""",
    "tests/test_cart.py": """\
import pytest

from shop.cart import Cart
from shop.errors import NotFoundError, ValidationError
from shop.models import Product

TEA = Product("tea", "Чай", 25000)
CUP = Product("cup", "Кружка", 40000)


def test_add_and_total():
    cart = Cart()
    cart.add(TEA, 2)
    cart.add(CUP)
    assert cart.count() == 3
    assert cart.total() == 2 * 25000 + 40000


def test_limit():
    cart = Cart(max_items=2)
    cart.add(TEA, 2)
    with pytest.raises(ValidationError):
        cart.add(CUP)


def test_remove_missing():
    with pytest.raises(NotFoundError):
        Cart().remove("nope")
""",
}


def make_project(root: Path) -> Path:
    """Создаёт учебный проект в ``root`` (и git-репозиторий с первым коммитом)."""
    for rel, text in FIXTURE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git_init(root)
    return root


def _git_init(root: Path) -> None:
    git = shutil.which("git")
    if git is None:
        return
    identity = ["-c", "user.name=gigacrew-eval", "-c", "user.email=eval@example.invalid"]
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        [*identity, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "shop: исходная версия"],
    ):
        done = subprocess.run([git, *args], cwd=root, capture_output=True, timeout=60)
        if done.returncode != 0:
            return


def build_index(root: Path) -> None:
    """Индекс проекта — как перед первым вводом в REPL (``ensure_index``)."""
    index = ProjectIndex(Workspace(root))
    try:
        index.refresh()
    except INDEX_ERRORS:
        pass  # без индекса навигация работает медленнее, но работает
    finally:
        index.close()


def run_fixture_tests(root: Path, timeout: int = 300) -> bool:
    """Проходят ли тесты учебного проекта после правок модели."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=root,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


_SNAPSHOT_SKIP = frozenset({".git", ".gigacrew", "__pycache__", ".pytest_cache"})


def snapshot(root: Path) -> dict[str, str]:
    """Содержимое файлов проекта (хеши) без служебных папок — для поиска изменений."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _SNAPSHOT_SKIP for part in rel.parts) or not path.is_file():
            continue
        files[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def changed_files(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """Добавленные, изменённые и удалённые файлы."""
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


# --------------------------------------------------------------------------- #
# Сценарный пользователь и запись обращений к модели
# --------------------------------------------------------------------------- #
_SAFE_COMMANDS = frozenset(
    {
        "cd",
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "rg",
        "find",
        "tree",
        "pwd",
        "echo",
        "sort",
        "diff",
        "pytest",
    }
)
_SAFE_GIT = frozenset({"status", "log", "diff", "show", "grep", "ls-files", "blame"})
# Перенаправления, подстановки и ключи, которыми «читающие» команды пишут файлы
# (find -exec/-delete/-fprint, git … --output; sort/tree -o — ниже).
_UNSAFE_SHELL = re.compile(r"[<>`]|\$\(|\s-(?:exec|execdir|delete|fprint\w*|fls)\b|--output")
_ENV_ASSIGNMENT = re.compile(r"\w+=\S*")


def is_safe_command(command: str) -> bool:
    """Команда только читает или гоняет тесты — сценарный пользователь её подтверждает.

    Каждая часть цепочки (``&&``, ``||``, ``;``, ``|``) — из короткого списка, без
    перенаправлений и подстановок; python — только ``-m pytest``, git — только чтение.
    """
    if _UNSAFE_SHELL.search(command):
        return False
    for segment in re.split(r"&&|\|\||[;|\n]", command):
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        while words and _ENV_ASSIGNMENT.fullmatch(words[0]):
            words.pop(0)
        if not words:
            continue
        name, args = words[0], words[1:]
        if name in ("python", "python3"):
            if args[:2] != ["-m", "pytest"]:
                return False
        elif name == "git":
            if not args or args[0] not in _SAFE_GIT:
                return False
        elif name not in _SAFE_COMMANDS:
            return False
        elif name in ("sort", "tree") and any(arg.startswith("-o") for arg in args):
            return False
    return True


def is_plan_question(questions: Sequence[Question]) -> bool:
    """Вопрос об одобрении плана (exit_plan_mode), а не вопрос модели через ask_user."""
    return len(questions) == 1 and questions[0].text == PLAN_QUESTION.text


class ScriptedUser(RecordingEvents):
    """Пользователь по сценарию.

    ``plan_answers`` — ответы на показы плана по порядку (``None`` — Esc, план не
    одобрен); когда ответы кончились — тоже Esc. На вопросы модели (``ask_user``) —
    первый вариант. Подтверждает правки файлов и безопасные команды, остальное
    отклоняет. ``on_plan`` вызывается при каждом показе плана (до ответа).
    """

    def __init__(
        self,
        plan_answers: Iterable[Answer | None] = (),
        *,
        on_plan: Callable[[], None] | None = None,
    ):
        super().__init__(confirm_answer=False)
        self._plan_answers = list(plan_answers)
        self._on_plan = on_plan
        self.plans: list[str] = []  # тексты показанных планов
        self.model_questions = 0  # вопросов модели через ask_user
        self.commands: list[tuple[str, bool]] = []  # (команда, подтверждена)

    def on_stream_delta(self, text: str) -> None:
        """Куски потока не копятся: итоговый текст есть в истории."""

    def ask_user(self, questions):
        questions = list(questions)
        self.questions.append(questions)
        if is_plan_question(questions):
            self.plans.append(questions[0].body)
            if self._on_plan is not None:
                self._on_plan()
            answer = self._plan_answers.pop(0) if self._plan_answers else None
            return None if answer is None else [answer]
        self.model_questions += len(questions)
        return [_first_option(q) for q in questions]

    def confirm(self, call, preview, *, dangerous):
        self.confirms.append((call, preview, dangerous))
        if dangerous:
            return False
        if call.kind is ToolKind.EDIT:
            return True
        if call.kind is ToolKind.COMMAND:
            # В описании вызова команда обрезана — полная только в превью.
            command = preview.text.removeprefix("$ ") if preview is not None else call.summary
            allowed = is_safe_command(command)
            self.commands.append((command, allowed))
            return allowed
        return False


def _first_option(question: Question) -> Answer:
    if question.options:
        return Answer((question.options[0].label,))
    return Answer(custom="Как считаешь нужным.")


class RecordingProvider(LLMProvider):
    """Обёртка провайдера: записывает каждое обращение к модели (для метрик)."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, messages, tools=None, **kwargs) -> AssistantTurn:
        started = time.monotonic()
        turn = self.inner.complete(messages, tools, **kwargs)
        self._record(messages, tools, kwargs, turn, started)
        return turn

    def stream(self, messages, tools=None, *, on_delta=None, **kwargs) -> AssistantTurn:
        started = time.monotonic()
        turn = self.inner.stream(messages, tools, on_delta=on_delta, **kwargs)
        self._record(messages, tools, kwargs, turn, started)
        return turn

    def _record(
        self,
        messages: Sequence[Message],
        tools: Any,
        kwargs: Mapping[str, Any],
        turn: AssistantTurn,
        started: float,
    ) -> None:
        system = messages[0].content if messages and messages[0].role == "system" else ""
        usage = turn.usage
        self.calls.append(
            {
                "subagent": system.startswith(_SUBAGENT_MARK),
                "summary": tools is None,  # сжатие контекста
                "messages": len(messages),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": usage.precached_prompt_tokens,
                "calls": [call.name for call in turn.message.function_calls],
                "finish_reason": turn.finish_reason,
                "reasoning_chars": len(turn.message.reasoning),
                "text": bool(turn.message.content.strip()),
                "tool_choice": kwargs.get("tool_choice", "auto"),
                "duration_s": round(time.monotonic() - started, 2),
            }
        )


# --------------------------------------------------------------------------- #
# Сценарии
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Turn:
    """Запрос пользователя; ``mode`` — режим, на который он переключается перед запросом."""

    text: str
    mode: PermissionMode | None = None


@dataclass(frozen=True)
class Scenario:
    """Сценарий: запросы, ответы на показы плана и критерий успеха.

    ``plan_only`` — режим плана на всём сценарии: проект меняться не должен (инвариант).
    ``expect`` — подстроки, которые должны быть в итоговом ответе; ``files_expect`` —
    подстроки в файлах после прогона; ``run_tests`` — прогнать тесты учебного проекта.
    """

    id: str
    goal: str
    mode: PermissionMode
    turns: tuple[Turn, ...]
    check: Callable[[Mapping[str, Any]], bool]
    plan_answers: tuple[Answer | None, ...] = ()
    expect: tuple[str, ...] = ()
    files_expect: Mapping[str, str] = field(default_factory=dict)
    plan_only: bool = False
    run_tests: bool = False


PROMO_TASK = (
    "Добавь промокоды: скидка в процентах при оформлении заказа, не больше предела скидки "
    "из настроек; неизвестный промокод — ошибка валидации."
)
DETAILED_PLAN = (
    "Нужен подробный план: для каждого шага — полный код изменений (функции и классы "
    "целиком, не фрагменты) и тесты к нему."
)
CLEAR_TASK = "добавь в Cart метод clear(), очищающий корзину, и тест к нему в tests/test_cart.py."
PLAN_REMARKS = "Добавь в план тесты на промокоды и проверку, что скидка не превышает предел."

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="plan_exit",
        goal="план показан пользователю (exit_plan_mode)",
        mode=PermissionMode.PLAN,
        turns=(Turn(PROMO_TASK),),
        plan_answers=(None,),
        plan_only=True,
        check=lambda r: r["plan_shown"] >= 1,
    ),
    Scenario(
        id="plan_detailed",
        goal="подробный план с кодом (длинный write_file) показан пользователю",
        mode=PermissionMode.PLAN,
        turns=(Turn(f"{PROMO_TASK} {DETAILED_PLAN}"),),
        plan_answers=(None,),
        plan_only=True,
        check=lambda r: r["plan_shown"] >= 1,
    ),
    Scenario(
        id="plan_remarks",
        goal="после замечаний план исправлен и показан снова",
        mode=PermissionMode.PLAN,
        turns=(Turn(PROMO_TASK),),
        plan_answers=(Answer((), custom=PLAN_REMARKS), None),
        plan_only=True,
        check=lambda r: r["plan_shown"] >= 2 and r["plan_revised"],
    ),
    Scenario(
        id="plan_implement",
        goal="одобренный план выполнен в том же ходе",
        mode=PermissionMode.PLAN,
        turns=(Turn(CLEAR_TASK[0].upper() + CLEAR_TASK[1:]),),
        plan_answers=(Answer((APPROVE_EDITS,)),),
        files_expect={"shop/cart.py": "def clear"},
        run_tests=True,
        check=lambda r: r["plan_shown"] >= 1 and bool(r["files_ok"]),
    ),
    Scenario(
        id="plan_manual_exit",
        goal="после ручного выхода из плана — правки без exit_plan_mode",
        mode=PermissionMode.PLAN,
        turns=(
            Turn(f"Спланируй: {CLEAR_TASK}"),
            Turn("Выполни план.", mode=PermissionMode.ACCEPT_EDITS),
        ),
        plan_answers=(None,),
        files_expect={"shop/cart.py": "def clear"},
        run_tests=True,
        check=lambda r: bool(r["files_ok"]) and r["turns"][-1]["exit_calls"] == 0,
    ),
    Scenario(
        id="plan_question",
        goal="на вопрос в режиме плана — ответ без плана",
        mode=PermissionMode.PLAN,
        turns=(Turn("Сколько товаров можно положить в корзину и где это проверяется?"),),
        expect=("50", "cart.py"),
        plan_only=True,
        check=lambda r: r["plan_shown"] == 0 and not r["plan_written"] and r["answer_hits"] == 1.0,
    ),
    Scenario(
        id="delegate_broad",
        goal="широкий вопрос — разведка через explore",
        mode=PermissionMode.MANUAL,
        turns=(
            Turn(
                "Как в проекте устроена обработка ошибок: где какие исключения возникают и "
                "как они превращаются в HTTP-ответы? Со ссылками path:line."
            ),
        ),
        expect=("errors.py", "api.py", "storage.py", "payments.py"),
        check=lambda r: r["explore_calls"] >= 1,
    ),
    Scenario(
        id="delegate_pinpoint",
        goal="точечный вопрос — без суб-агентов",
        mode=PermissionMode.MANUAL,
        turns=(Turn("Какое исключение бросает parse_price из shop/money.py на пустую строку?"),),
        expect=("ValidationError",),
        check=lambda r: r["task_calls"] == 0 and r["answer_hits"] == 1.0,
    ),
    Scenario(
        id="plan_explore",
        goal="в режиме плана — разведка через explore",
        mode=PermissionMode.PLAN,
        turns=(
            Turn(
                "Спланируй перевод хранилища заказов с JSON-файла на SQLite: что и где "
                "придётся изменить."
            ),
        ),
        plan_answers=(None,),
        plan_only=True,
        check=lambda r: r["explore_calls"] >= 1,
    ),
    Scenario(
        id="delegate_subtasks",
        goal="две независимые подзадачи — суб-агенты",
        mode=PermissionMode.ACCEPT_EDITS,
        turns=(
            Turn(
                "Напиши тесты для shop/payments.py (в tests/test_payments.py) и для "
                "shop/storage.py (в tests/test_storage.py)."
            ),
        ),
        files_expect={"tests/test_payments.py": "def test", "tests/test_storage.py": "def test"},
        run_tests=True,
        check=lambda r: r["task_calls"] >= 1 and bool(r["files_ok"]),
    ),
)
SCENARIO_BY_ID: dict[str, Scenario] = {scenario.id: scenario for scenario in SCENARIOS}


# --------------------------------------------------------------------------- #
# Прогон и метрики
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    """Итог прогона: запись для отчёта, история для разбора и сбой (если был)."""

    record: dict[str, Any]
    transcript: dict[str, Any]
    error: BaseException | None = None


def run_scenario(
    scenario: Scenario,
    *,
    provider: LLMProvider,
    config: Config,
    windows: ModelWindows | None = None,
    chat_id: str | None = None,
) -> ScenarioResult:
    """Прогоняет сценарий на проекте ``config.project_root`` и собирает метрики.

    Сбой (ошибка API, таймаут, Ctrl+C) не теряет записи: он сохраняется в
    ``ScenarioResult.error``, пробросить его — дело вызывающего.
    """
    root = config.project_root
    build_index(root)
    before = snapshot(root)
    at_plan: list[dict[str, str]] = []  # файлы проекта на момент первого показа плана

    def on_plan() -> None:
        if not at_plan:
            at_plan.append(snapshot(root))

    recorder = RecordingProvider(provider)
    user = ScriptedUser(scenario.plan_answers, on_plan=on_plan)
    agent = Agent(
        recorder, build_default_registry(), config, user, windows=windows, chat_id=chat_id
    )
    turns: list[dict[str, Any]] = []
    error: BaseException | None = None
    started = time.monotonic()
    try:
        for turn in scenario.turns:
            if turn.mode is not None:
                agent.set_mode(turn.mode)
            turns.append(_turn_start(agent, user))
            agent.run_turn(turn.text)
            _turn_end(turns[-1], agent, user)
    except BaseException as e:  # запись нужна и при сбое; пробрасывает вызывающий
        error = e
        if turns and "mode_after" not in turns[-1]:
            _turn_end(turns[-1], agent, user)
    duration = time.monotonic() - started

    after = snapshot(root)
    record = _metrics(scenario, agent, user, recorder, turns, root, before, after, at_plan)
    record.update(
        scenario=scenario.id,
        model=config.model,
        temperature=config.temperature,
        window=windows.get(config.model) if windows is not None else None,
        duration_s=round(duration, 1),
        error=None if error is None else f"{type(error).__name__}: {error}",
    )
    if scenario.run_tests and error is None:
        record["tests_pass"] = run_fixture_tests(root)
    record["ok"] = error is None and bool(scenario.check(record))
    transcript = {
        "conversation": agent.conversation.to_dict(),
        "plans": user.plans,
        "notices": [list(notice) for notice in user.notices],
        "commands": [list(command) for command in user.commands],
        "requests": recorder.calls,
    }
    return ScenarioResult(record=record, transcript=transcript, error=error)


def _turn_start(agent: Agent, user: ScriptedUser) -> dict[str, Any]:
    return {
        "mode_before": agent.mode.value,
        "start": len(agent.conversation),
        "plans_before": len(user.plans),
    }


def _turn_end(info: dict[str, Any], agent: Agent, user: ScriptedUser) -> None:
    messages = agent.conversation.messages[info.pop("start") :]
    last = messages[-1] if messages else None
    answer = next(
        (
            m.content
            for m in reversed(messages)
            if m.role == "assistant" and not m.function_calls and m.content.strip()
        ),
        "",
    )
    stats = agent.last_turn
    info.update(
        mode_after=agent.mode.value,
        plan_shown=len(user.plans) - info.pop("plans_before"),
        exit_calls=sum(1 for call in _calls(messages) if call.name == "exit_plan_mode"),
        ended_in_text=bool(
            last is not None
            and last.role == "assistant"
            and not last.function_calls
            and last.content.strip()
        ),
        answer=answer[:MAX_ANSWER_CHARS],
        steps=stats.steps if stats is not None else None,
        stop_reason=stats.stop_reason if stats is not None else None,
    )


def _calls(messages: Iterable[Message]) -> Iterator[FunctionCall]:
    for message in messages:
        if message.role == "assistant":
            yield from message.function_calls


def _subagent_type(call: FunctionCall) -> str:
    """Тип суб-агента из вызова task (имя параметра — и прежнее, и будущее)."""
    value = call.arguments.get("subagent_type") or call.arguments.get("agent") or ""
    return str(value).strip().lower()


def _targets(call: FunctionCall, root: Path, target: Path | None) -> bool:
    """Вызов работает с файлом ``target`` (путь в аргументах — относительный или нет)."""
    raw = call.arguments.get("path")
    if target is None or not isinstance(raw, str) or not raw.strip():
        return False
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve() == target.resolve()
    except OSError:
        return False


def well_formed(messages: Sequence[Message]) -> bool:
    """Каждый вызов инструмента получил ровно один результат, и результат — после вызова."""
    pending = 0
    for message in messages:
        if message.role == "function":
            if pending == 0:
                return False
            pending -= 1
            continue
        if pending:
            return False
        if message.role == "assistant":
            pending = len(message.function_calls)
    return pending == 0


def _metrics(
    scenario: Scenario,
    agent: Agent,
    user: ScriptedUser,
    recorder: RecordingProvider,
    turns: list[dict[str, Any]],
    root: Path,
    before: Mapping[str, str],
    after: Mapping[str, str],
    at_plan: list[dict[str, str]],
) -> dict[str, Any]:
    messages = agent.conversation.messages
    calls = list(_calls(messages))
    tasks = [call for call in calls if call.name == "task"]
    explores = sum(1 for call in calls if call.name == "explore")
    plan_path = agent.plan_path
    try:
        plan_text = plan_path.read_text(encoding="utf-8").strip() if plan_path else ""
    except OSError:
        plan_text = ""
    results = [m.content for m in messages if m.role == "function"]
    answer = turns[-1].get("answer", "") if turns else ""
    hits = None
    if scenario.expect:
        found = sum(1 for item in scenario.expect if item.lower() in answer.lower())
        hits = round(found / len(scenario.expect), 2)
    files_ok = None
    if scenario.files_expect:
        files_ok = all(
            (root / rel).is_file() and text in (root / rel).read_text(encoding="utf-8")
            for rel, text in scenario.files_expect.items()
        )
    subagents = [info for kind, info in user.events if kind == "subagent_start"]
    main_requests = [c for c in recorder.calls if not c["subagent"] and not c["summary"]]
    return {
        # план
        "plan_shown": len(user.plans),
        "plan_revised": len(user.plans) >= 2 and user.plans[-1] != user.plans[0],
        "plan_written": bool(plan_text),
        "plan_chars": len(plan_text),
        "exit_calls": sum(1 for call in calls if call.name == "exit_plan_mode"),
        "plan_writes": sum(
            1 for c in calls if c.name == "write_file" and _targets(c, root, plan_path)
        ),
        "plan_edits": sum(
            1 for c in calls if c.name == "edit_file" and _targets(c, root, plan_path)
        ),
        "plan_reminders": sum(
            1 for _, text in user.notices if any(mark in text for mark in HARNESS_PLAN_NOTICES)
        ),
        # самопроверки и те из них, после которых модель продолжила работу (служебное
        # сообщение осталось в истории); остальные кончились ответом «OK»
        "self_checks": sum(s.self_checks for s in user.stats),
        "self_checks_resumed": sum(1 for m in messages if m.meta and SELF_CHECK_MARK in m.content),
        "plan_turns_ended_in_text": sum(
            1
            for t in turns
            if t["mode_before"] == PermissionMode.PLAN.value
            and t.get("mode_after") == PermissionMode.PLAN.value
            and t.get("plan_shown") == 0
            and t.get("ended_in_text")
        ),
        "blocked_calls": sum(1 for text in results if text.startswith(PLAN_BLOCKED_NOTE)),
        "final_mode": agent.mode.value,
        # делегирование: запуски суб-агентов (task и explore) и из них — разведка
        "task_calls": len(tasks) + explores,
        "explore_calls": explores + sum(1 for call in tasks if _subagent_type(call) == "explore"),
        "general_calls": sum(
            1 for call in tasks if _subagent_type(call) in ("general", "general-purpose")
        ),
        "task_batches": sum(
            1
            for m in messages
            if m.role == "assistant"
            and sum(c.name in ("task", "explore") for c in m.function_calls) >= 2
        ),
        "subagent_runs": len(subagents),
        "subagent_tool_calls": sum(1 for kind, _ in user.events if kind == "subagent_tool_call"),
        "research_calls": sum(1 for call in calls if call.name in RESEARCH_TOOLS),
        "ask_user_calls": sum(1 for call in calls if call.name == "ask_user"),
        "model_questions": user.model_questions,
        # команды и отказы
        "commands": len(user.commands),
        "commands_rejected": sum(1 for _, allowed in user.commands if not allowed),
        "rejected_calls": sum(1 for text in results if text.startswith(REJECTED_NOTE)),
        # расход
        "requests": len(recorder.calls),
        "requests_subagent": sum(1 for c in recorder.calls if c["subagent"]),
        "summaries": sum(1 for c in recorder.calls if c["summary"]),
        "prompt_tokens": sum(c["prompt_tokens"] for c in recorder.calls),
        "completion_tokens": sum(c["completion_tokens"] for c in recorder.calls),
        "cached_tokens": sum(c["cached_tokens"] for c in recorder.calls),
        "text_with_calls": sum(1 for c in main_requests if c["text"] and c["calls"]),
        "cut_retries": sum(s.cut_retries for s in user.stats),
        "rate_limit_waits": sum(1 for _, text in user.notices if RATE_LIMIT_MARK in text),
        "max_completion_tokens": max((c["completion_tokens"] for c in recorder.calls), default=0),
        "max_request_s": max((c["duration_s"] for c in recorder.calls), default=0.0),
        "steps": sum(s.steps for s in user.stats),
        "tool_calls": sum(s.tool_calls for s in user.stats),
        "stop_reasons": [s.stop_reason for s in user.stats if s.stop_reason],
        # качество
        "answer": answer[:MAX_ANSWER_CHARS],
        "answer_hits": hits,
        "changed_files": changed_files(before, after),
        "changed_before_plan": changed_files(before, at_plan[0]) if at_plan else [],
        "files_ok": files_ok,
        "tests_pass": None,
        "reminder_leaks": sum(
            1 for m in messages if m.role == "assistant" and "system-reminder" in m.content
        ),
        "well_formed": well_formed(messages) and not agent.conversation.pending_calls(),
        "turns": turns,
    }


def invariant_errors(scenario: Scenario, record: Mapping[str, Any]) -> list[str]:
    """Нарушения инвариантов harness'а (не поведения модели) — тест падает только на них."""
    errors = []
    if not record["well_formed"]:
        errors.append("история нарушена: результат без вызова или вызов без результата")
    if scenario.plan_only and record["changed_files"]:
        errors.append(f"в режиме плана изменены файлы проекта: {record['changed_files']}")
    if record["changed_before_plan"]:
        errors.append(f"до показа плана изменены файлы проекта: {record['changed_before_plan']}")
    return errors


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_") or "x"


def git_revision(root: Path = REPO_ROOT) -> str:
    """Коммит репозитория (с пометкой о незакоммиченных правках) — для метаданных отчёта."""
    git = shutil.which("git")
    if git is None:
        return ""
    try:
        sha = subprocess.run(
            [git, "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            [git, "status", "--porcelain"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""
    return f"{sha}-dirty" if sha and dirty else sha


@dataclass
class EvalSession:
    """Сессия eval: общий конфиг и провайдер, записи прогонов и файл отчёта.

    Отчёт переписывается после каждого прогона — сорвавшаяся сессия не теряет готовое.
    Истории прогонов — рядом, в папке с именем отчёта.
    """

    config: Config
    provider: LLMProvider
    windows: ModelWindows
    directory: Path
    models: list[str]
    runs: int
    label: str = ""
    stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"eval-{self.stamp}" + (f"-{_slug(self.label)}" if self.label else "")

    @property
    def path(self) -> Path:
        return self.directory / f"{self.name}.json"

    def config_for(self, model: str, scenario: Scenario, root: Path) -> Config:
        """Конфиг прогона: как в REPL, но без сохранения чатов, замеров и ``-y``."""
        return replace(
            self.config,
            project_root=root,
            model=model,
            mode=scenario.mode,
            stream=True,
            save_chats=False,
            auto_measure=False,
            auto_approve=False,
            yes_all=False,
        )

    def add(self, result: ScenarioResult, *, run: int) -> dict[str, Any]:
        record = {**result.record, "run": run}
        transcript = (
            self.directory
            / self.name
            / f"{_slug(record['model'])}-{record['scenario']}-r{run}.json"
        )
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            json.dumps(result.transcript, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        record["transcript"] = transcript.relative_to(self.directory).as_posix()
        self.records.append(record)
        self.write()
        return record

    def meta(self) -> dict[str, Any]:
        return {
            "created": self.stamp,
            "label": self.label,
            "git": git_revision(),
            "temperature": self.config.temperature,
            "reasoning": self.config.reasoning,
            "models": self.models,
            "runs": self.runs,
            "windows": {model: self.windows.get(model) for model in self.models},
            "scenarios": {s.id: s.goal for s in SCENARIOS},
        }

    def write(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        report = {"meta": self.meta(), "runs": self.records}
        self.path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.path


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


Metric = str | Callable[[Mapping[str, Any]], float]


def _value(record: Mapping[str, Any], metric: Metric) -> float:
    return float(metric(record) if callable(metric) else record.get(metric) or 0)


def _rate(records: Sequence[Mapping[str, Any]], metric: Metric) -> str:
    """Доля прогонов, где метрика истинна: «2/3» («—» — прогонов нет)."""
    if not records:
        return "—"
    return f"{sum(1 for r in records if _value(r, metric))}/{len(records)}"


def _mean(records: Sequence[Mapping[str, Any]], metric: Metric, digits: int = 1) -> str:
    if not records:
        return "—"
    value = sum(_value(r, metric) for r in records) / len(records)
    return f"{value:.{digits}f}"


def _cache(records: Sequence[Mapping[str, Any]]) -> str:
    prompt = sum(r.get("prompt_tokens") or 0 for r in records)
    cached = sum(r.get("cached_tokens") or 0 for r in records)
    return f"{100 * cached / prompt:.0f}%" if prompt else "—"


def _tokens(record: Mapping[str, Any]) -> float:
    return ((record.get("prompt_tokens") or 0) + (record.get("completion_tokens") or 0)) / 1000


def _groups(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list]:
    groups: dict[tuple[str, str], list] = {}
    for record in records:
        groups.setdefault((record["model"], record["scenario"]), []).append(record)
    return groups


def _ordered(keys: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    order = {scenario.id: i for i, scenario in enumerate(SCENARIOS)}
    return sorted(keys, key=lambda key: (key[0], order.get(key[1], len(order)), key[1]))


def _table(rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


SUMMARY_LEGEND = (
    "ok — сценарий выполнен (цели ниже); err — сорвавшиеся прогоны; exit — модель сама "
    "вызвала exit_plan_mode; shown — план показан пользователю; rem — план записан, но не "
    "отправлен моделью (напоминание, с этапа 3 — показ harness'ом); chk — самопроверок; "
    "txt — ходов плана, закончившихся текстом без показа плана (в среднем); task/expl — "
    "запусков суб-агентов (task и explore) / из них разведка; res — своих поисков и "
    "чтений; req — запросов к модели "
    "(с суб-агентами); tok,k — тысяч токенов; cache — доля токенов запроса из кеша; time — "
    "секунд."
)


def summary_table(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Таблица по модели и сценарию: доли успехов и средние по повторам."""
    groups = _groups(records)
    rows: list[tuple[str, ...]] = [
        (
            "model",
            "scenario",
            "ok",
            "err",
            "exit",
            "shown",
            "rem",
            "chk",
            "txt",
            "task",
            "expl",
            "res",
            "req",
            "tok,k",
            "cache",
            "time",
        )
    ]
    for key in _ordered(groups):
        runs = groups[key]
        rows.append(
            (
                key[0],
                key[1],
                _rate(runs, "ok"),
                str(sum(1 for r in runs if r.get("error"))),
                _rate(runs, lambda r: r["exit_calls"] > 0),
                _rate(runs, lambda r: r["plan_shown"] > 0),
                _mean(runs, "plan_reminders"),
                _mean(runs, "self_checks"),
                _mean(runs, "plan_turns_ended_in_text"),
                _mean(runs, "task_calls"),
                _mean(runs, "explore_calls"),
                _mean(runs, "research_calls"),
                _mean(runs, "requests"),
                _mean(runs, _tokens),
                _cache(runs),
                _mean(runs, "duration_s", digits=0),
            )
        )
    goals = [f"  {scenario.id} — {scenario.goal}" for scenario in SCENARIOS]
    return [*_table(rows), "", SUMMARY_LEGEND, "Цели сценариев:", *goals]


def _describe(meta: Mapping[str, Any]) -> str:
    parts = [str(meta.get("created", "?"))]
    for key in ("label", "git"):
        if meta.get(key):
            parts.append(str(meta[key]))
    parts.append(f"T={meta.get('temperature')}")
    if meta.get("reasoning") not in (None, "off"):
        parts.append(f"reasoning={meta['reasoning']}")
    return " · ".join(parts)


def _errors(records: Sequence[Mapping[str, Any]]) -> str:
    return str(sum(1 for r in records if r.get("error")))


# Столбцы сравнения: имя и значение по группе прогонов.
_COMPARED: tuple[tuple[str, Callable[[Sequence[Mapping[str, Any]]], str]], ...] = (
    ("ok", lambda g: _rate(g, "ok")),
    ("err", _errors),
    ("exit", lambda g: _rate(g, lambda r: r["exit_calls"] > 0)),
    ("shown", lambda g: _rate(g, lambda r: r["plan_shown"] > 0)),
    ("rem", lambda g: _mean(g, "plan_reminders")),
    ("chk", lambda g: _mean(g, "self_checks")),
    ("txt", lambda g: _mean(g, "plan_turns_ended_in_text")),
    ("expl", lambda g: _mean(g, "explore_calls")),
    ("task", lambda g: _mean(g, "task_calls")),
    ("res", lambda g: _mean(g, "research_calls")),
    ("req", lambda g: _mean(g, "requests")),
    ("tok,k", lambda g: _mean(g, _tokens)),
)


def compare_reports(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Сравнение двух отчётов по модели и сценарию: «было → стало»."""
    before, after = _groups(old["runs"]), _groups(new["runs"])
    rows: list[tuple[str, ...]] = [("model", "scenario", *(name for name, _ in _COMPARED))]
    for key in _ordered(before.keys() | after.keys()):
        a, b = before.get(key, []), after.get(key, [])
        rows.append((key[0], key[1], *(f"{value(a)} → {value(b)}" for _, value in _COMPARED)))
    head = [f"было:  {_describe(old['meta'])}", f"стало: {_describe(new['meta'])}", ""]
    return head + _table(rows)


# --------------------------------------------------------------------------- #
# Выжимка для чата
# --------------------------------------------------------------------------- #
DIGEST_PROBLEMS = 30  # неуспешных прогонов в выжимке, остальные — числом
DIGEST_ERRORS = 5  # частых ошибок
_CLIP = 100  # символов текста ошибки или ответа


def _clip(text: str, limit: int = _CLIP) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _peak(records: Sequence[Mapping[str, Any]], metric: str, scale: float = 1.0) -> str:
    """Максимум по прогонам («—» — метрики нет: отчёт старой версии)."""
    values = [r[metric] for r in records if r.get(metric) is not None]
    return f"{max(values) / scale:.1f}" if values else "—"


def _stops(records: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(reason for r in records for reason in r.get("stop_reasons") or ())
    return ",".join(f"{reason}×{n}" if n > 1 else reason for reason, n in counts.most_common())


def _problem(record: Mapping[str, Any]) -> str:
    """Почему прогон не удался — одной строкой."""
    parts = [f"{record['model']} {record['scenario']} r{record.get('run', '?')}:"]
    if record.get("error"):
        parts.append(f"сбой «{_clip(record['error'])}»")
    if record.get("stop_reasons"):
        parts.append("стоп " + ",".join(record["stop_reasons"]))
    parts.append(
        f"shown={record.get('plan_shown', 0)} exit={record.get('exit_calls', 0)} "
        f"chk={record.get('self_checks', 0)} task={record.get('task_calls', 0)}"
    )
    if record.get("answer"):
        parts.append(f"ответ «{_clip(record['answer'], 60)}»")
    return " ".join(parts)


def digest_report(report: Mapping[str, Any]) -> list[str]:
    """Выжимка отчёта, которую можно целиком вставить в чат (до ~100 строк).

    Сводка по модели и сценарию — с пиком вывода и длительности запроса, причинами
    остановок, — затем неуспешные прогоны по одному в строку (сначала без сбоя) и частые
    ошибки.
    """
    runs = report["runs"]
    groups = _groups(runs)
    rows: list[tuple[str, ...]] = [
        (
            "model",
            "scenario",
            "ok",
            "err",
            "rem",
            "chk",
            "txt",
            "res",
            "req",
            "cut",
            "out,k",
            "max,s",
            "stops",
        )
    ]
    for key in _ordered(groups):
        g = groups[key]
        rows.append(
            (
                key[0],
                key[1],
                _rate(g, "ok"),
                _errors(g),
                _mean(g, "plan_reminders"),
                _mean(g, "self_checks"),
                _mean(g, "plan_turns_ended_in_text"),
                _mean(g, "research_calls", digits=0),
                _mean(g, "requests", digits=0),
                str(sum(r.get("cut_retries") or 0 for r in g)),
                _peak(g, "max_completion_tokens", scale=1000),
                _peak(g, "max_request_s"),
                _stops(g),
            )
        )
    lines = [_describe(report["meta"]), *_table(rows)]
    # Сначала провалы поведения, потом сбои: сбои ещё и сведены в «Частые ошибки» ниже,
    # и сотня одинаковых таймаутов не должна вытеснить единичный провал.
    failed = sorted((r for r in runs if not r.get("ok")), key=lambda r: bool(r.get("error")))
    if failed:
        lines += ["", f"Неуспешные прогоны: {len(failed)} из {len(runs)}"]
        lines += [_problem(r) for r in failed[:DIGEST_PROBLEMS]]
        if len(failed) > DIGEST_PROBLEMS:
            lines.append(f"… и ещё {len(failed) - DIGEST_PROBLEMS}")
    waits = [r.get("rate_limit_waits") or 0 for r in runs]
    if any(waits):
        lines += [
            "",
            f"Пауз из-за лимита запросов (429): {sum(waits)} в {sum(1 for w in waits if w)} прогонах",
        ]
    errors = Counter(_clip(r["error"]) for r in runs if r.get("error"))
    if errors:
        lines += ["", "Частые ошибки:"]
        lines += [f"{n}× {text}" for text, n in errors.most_common(DIGEST_ERRORS)]
    lines += [
        "",
        "rem — план не отправлен моделью; chk — самопроверок; txt — ходов плана, кончившихся "
        "текстом; cut — повторов после оборванного ответа; out,k — пик вывода за запрос, "
        "тыс. токенов; max,s — самый долгий запрос.",
    ]
    return lines


# --------------------------------------------------------------------------- #
# Пробы шлюза (test_api_probe_live.py)
# --------------------------------------------------------------------------- #
def reasoning_texts(data: Mapping[str, Any]) -> list[str]:
    """Рассуждения модели из ответа API v2 (сообщения ``role: "reasoning"``)."""
    texts: list[str] = []
    for message in data.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "reasoning":
            continue
        content = message.get("content")
        parts = [{"text": content}] if isinstance(content, str) else content or []
        for part in parts:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return [text for text in texts if text.strip()]


def with_reasoning(payload: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Копия запроса с рассуждениями (``role: "reasoning"``) перед последним ответом
    ассистента — тем же сообщением, каким API отдаёт их в ответе."""
    copy = json.loads(json.dumps(payload))
    messages = copy["messages"]
    index = max(i for i, message in enumerate(messages) if message["role"] == "assistant")
    messages.insert(index, {"role": "reasoning", "content": [{"text": text}]})
    return copy


def _status(step: Mapping[str, Any] | None, base: int | None = None) -> str:
    """Исход запроса пробы: размер запроса (и прирост к ``base``) или код ошибки."""
    if step is None:
        return "—"
    if step["status"] >= 400:
        return f"ошибка {step['status']}"
    tokens = step["input_tokens"]
    if base and tokens is not None:
        return f"{tokens} ток. ({tokens - base:+d})"
    return f"{tokens} ток."


def describe_probe(model: str, finding: Mapping[str, Any]) -> list[str]:
    """Находки проб по одной модели — строками для вывода."""
    lines = [f"{model}:"]
    if "reasoning_chars" in finding:
        chars = finding["reasoning_chars"]
        returned = f"да, {chars} симв. (~{chars // CHARS_PER_TOKEN} ток.)" if chars else "нет"
        lines.append(f"  рассуждения в ответе с вызовом функции: {returned}")
    steps = finding.get("step2")
    if steps:
        lines.append(
            f"  второй шаг, input_tokens: с tools_state_id {_status(steps.get('with_state'))}, "
            f"без него {_status(steps.get('without_state'))}"
        )
        # Порог «заметной» разницы — половина оценки рассуждений, но не меньше 5 токенов.
        threshold = max(5, finding.get("reasoning_chars", 0) // CHARS_PER_TOKEN // 2)
        with_state = steps["with_state"].get("input_tokens")
        without = (steps.get("without_state") or {}).get("input_tokens")
        if with_state and without:
            delta = with_state - without
            verdict = (
                "похоже, шлюз сам подставляет скрытое состояние (рассуждения) по tools_state_id"
                if delta >= threshold
                else "признаков, что шлюз хранит рассуждения сам, нет"
            )
            lines.append(f"    разница {delta:+d} ток.: {verdict}")
        if "with_state+reasoning" in steps:
            lines.append(
                "  рассуждения в запросе (role: reasoning): с tools_state_id "
                f"{_status(steps['with_state+reasoning'], with_state)}, без него "
                f"{_status(steps.get('without_state+reasoning'), without)}"
            )
    elif finding.get("tool_call") is False:
        lines.append("  модель не вызвала функцию — шаги не сравнивались")
    sampling = finding.get("sampling")
    if sampling:
        verdicts = (
            f"{name} → {'принято' if step['status'] < 400 else 'ошибка ' + str(step['status'])}"
            for name, step in sampling.items()
        )
        lines.append(f"  model_options: {'; '.join(verdicts)} (принято ≠ учтено)")
    if "text_stream_finish" in finding:
        finish = finding["text_stream_finish"]
        verdict = (
            "окончание не приходит — каждый ответ сочтётся оборванным!"
            if finish == "incomplete"
            else "окончание приходит"
        )
        lines.append(f"  текстовый поток: finish_reason={finish} — {verdict}")
    cache = finding.get("cache")
    if cache and len(cache) >= 2:
        second = cache[-1]
        share = (
            f" ({100 * second['cached_tokens'] / second['input_tokens']:.0f}%)"
            if second["input_tokens"]
            else ""
        )
        lines.append(
            f"  кеш префикса: повтор запроса {second['input_tokens']} ток., из кеша "
            f"{second['cached_tokens']}{share}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "show":
        report = load_report(Path(args[1]))
        print(_describe(report["meta"]))
        print("\n".join(summary_table(report["runs"])))
        return 0
    if len(args) == 2 and args[0] == "digest":
        print("\n".join(digest_report(load_report(Path(args[1])))))
        return 0
    if len(args) == 3 and args[0] == "compare":
        lines = compare_reports(load_report(Path(args[1])), load_report(Path(args[2])))
        print("\n".join(lines))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
