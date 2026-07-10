"""Product catalog: hierarchical categories, variants, and tiered pricing.

A :class:`Product` is the merchandising unit a shopper browses (a "T-shirt");
a :class:`Variant` is the sellable, stock-keeping unit (a specific
size/color/material combination with its own SKU). Prices hang off the
variant so the same product can carry different prices per option, per
customer tier (retail / wholesale / member), and per currency.

Everything here is in-memory and cheap to construct; durable state (stock,
carts, orders) lives in SQLite via the other modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional

from .money import Money


class PriceTier(str, Enum):
    RETAIL = "retail"
    WHOLESALE = "wholesale"
    MEMBER = "member"


@dataclass
class Category:
    """A node in the category tree. ``parent`` is ``None`` for a root."""

    id: str
    name: str
    parent: Optional[str] = None


@dataclass
class Variant:
    """A sellable SKU: one concrete combination of option values.

    ``options`` holds the axis values, e.g. ``{"size": "M", "color": "red"}``.
    ``prices`` is keyed by ``(tier, currency)`` so a single variant can quote
    a member price in SAR and a retail price in USD independently.
    """

    sku: str
    options: Dict[str, str] = field(default_factory=dict)
    prices: Dict[tuple[PriceTier, str], Money] = field(default_factory=dict)
    active: bool = True  # False once discontinued; stays for order history

    def set_price(self, tier: PriceTier, price: Money) -> None:
        self.prices[(tier, price.currency)] = price

    def price(self, tier: PriceTier, currency: str) -> Money:
        """Best price for ``tier``, falling back to RETAIL then to any tier.

        Wholesale/member customers should never pay *more* than retail, so if
        a specific tier price is missing we fall back to retail rather than
        erroring, and only raise if the currency itself is unpriced.
        """
        key = (tier, currency)
        if key in self.prices:
            return self.prices[key]
        retail = (PriceTier.RETAIL, currency)
        if retail in self.prices:
            return self.prices[retail]
        for (t, c), m in self.prices.items():
            if c == currency:
                return m
        raise KeyError(f"{self.sku}: no price in {currency}")


@dataclass
class Product:
    """A merchandising unit grouping one or more variants.

    ``attributes`` are the filterable/facetable properties shown in search
    (brand, material, connector type…). ``tags`` and the title/description
    feed the full-text index.
    """

    id: str
    title: str
    category: str
    description: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    variants: List[Variant] = field(default_factory=list)
    popularity: int = 0  # a business signal for relevance ranking

    def variant(self, sku: str) -> Optional[Variant]:
        return next((v for v in self.variants if v.sku == sku), None)

    def active_variants(self) -> List[Variant]:
        return [v for v in self.variants if v.active]


class Catalog:
    """In-memory registry of categories, products, and variants.

    Provides the lookups the rest of the system needs: resolve a SKU to its
    variant/product, walk the category tree for hierarchical filtering, and
    enumerate variants for the search index.
    """

    def __init__(self) -> None:
        self._categories: Dict[str, Category] = {}
        self._products: Dict[str, Product] = {}
        self._sku_index: Dict[str, tuple[str, Variant]] = {}

    # ---- registration ----------------------------------------------------
    def add_category(self, category: Category) -> Category:
        if category.parent and category.parent not in self._categories:
            raise KeyError(f"unknown parent category {category.parent!r}")
        self._categories[category.id] = category
        return category

    def add_product(self, product: Product) -> Product:
        if product.category not in self._categories:
            raise KeyError(f"unknown category {product.category!r}")
        self._products[product.id] = product
        for v in product.variants:
            self._sku_index[v.sku] = (product.id, v)
        return product

    # ---- lookups ---------------------------------------------------------
    def product(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def resolve(self, sku: str) -> Optional[tuple[Product, Variant]]:
        hit = self._sku_index.get(sku)
        if hit is None:
            return None
        product_id, variant = hit
        return self._products[product_id], variant

    def variant(self, sku: str) -> Optional[Variant]:
        hit = self._sku_index.get(sku)
        return hit[1] if hit else None

    def is_sellable(self, sku: str) -> bool:
        """True only if the SKU exists and neither it nor its product is retired."""
        hit = self._sku_index.get(sku)
        return bool(hit and hit[1].active)

    def products(self) -> Iterator[Product]:
        return iter(self._products.values())

    def variants(self) -> Iterator[tuple[Product, Variant]]:
        for product in self._products.values():
            for variant in product.variants:
                yield product, variant

    # ---- category tree ---------------------------------------------------
    def category(self, category_id: str) -> Optional[Category]:
        return self._categories.get(category_id)

    def ancestors(self, category_id: str) -> List[str]:
        """Return ``category_id`` and every ancestor up to the root."""
        chain: List[str] = []
        current: Optional[str] = category_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            node = self._categories.get(current)
            current = node.parent if node else None
        return chain

    def descendants(self, category_id: str) -> List[str]:
        """Return ``category_id`` and every category beneath it (inclusive)."""
        children: Dict[str, List[str]] = {}
        for cat in self._categories.values():
            if cat.parent:
                children.setdefault(cat.parent, []).append(cat.id)
        out: List[str] = []
        stack = [category_id]
        while stack:
            cid = stack.pop()
            out.append(cid)
            stack.extend(children.get(cid, []))
        return out

    def discontinue(self, sku: str) -> None:
        """Retire a variant so carts drop it and it disappears from search."""
        hit = self._sku_index.get(sku)
        if hit:
            hit[1].active = False


def facet_values(products: Iterable[Product], attribute: str) -> Dict[str, int]:
    """Count how many products carry each value of ``attribute``.

    Powers faceted navigation: the sidebar of "Brand (12), Material (8)…"
    counts shown next to each filter option.
    """
    counts: Dict[str, int] = {}
    for p in products:
        value = p.attributes.get(attribute)
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return counts
