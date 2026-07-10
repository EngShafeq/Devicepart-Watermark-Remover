"""Search: relevance, facets, typo tolerance, synonyms, and the 200ms budget."""

import random

from ecommerce import Catalog, Category, Money, PriceTier, Product, SearchIndex, Variant


def build_index(catalog):
    return SearchIndex(
        catalog,
        synonyms={"display": ["screen", "display", "lcd"]},
        facet_attributes=["brand", "grade"],
    )


def test_full_text_finds_by_title(catalog):
    index = build_index(catalog)
    results = index.search("oled screen")
    assert results.hits
    assert results.hits[0].product_id == "p-oled-14"


def test_typo_tolerance_corrects_query(catalog):
    index = build_index(catalog)
    results = index.search("scren")  # missing a letter
    assert "scren" in results.corrected_terms
    assert results.corrected_terms["scren"] == "screen"
    assert any(h.product_id == "p-oled-14" for h in results.hits)


def test_synonym_expansion_matches(catalog):
    index = build_index(catalog)
    # "display" is a synonym of "screen"; the OLED product tags "display".
    results = index.search("lcd")
    assert any(h.product_id == "p-oled-14" for h in results.hits)


def test_faceted_navigation_counts(catalog):
    index = build_index(catalog)
    results = index.search("battery")
    facets = {f.attribute: f.values for f in results.facets}
    assert "brand" in facets
    assert facets["brand"].get("DevicePart", 0) >= 1


def test_attribute_filter(catalog):
    index = build_index(catalog)
    results = index.search("iphone", filters={"grade": "premium"})
    # Only the premium-grade OLED matches, not the standard battery.
    assert all(index.catalog.product(h.product_id).attributes["grade"] == "premium" for h in results.hits)


def test_category_scoping(catalog):
    index = build_index(catalog)
    results = index.search("iphone", category="batteries")
    assert {h.product_id for h in results.hits} == {"p-batt-14"}


def test_discontinued_products_excluded(catalog):
    catalog.discontinue("TOOLKIT")
    index = build_index(catalog)
    results = index.search("screwdriver kit")
    assert all(h.product_id != "p-tool" for h in results.hits)


def test_relevance_blends_text_and_popularity(catalog):
    # Two products both match "kit" in text; the more popular ranks first
    # only when text scores are comparable.
    index = build_index(catalog)
    results = index.search("repair toolkit kit")
    assert results.hits[0].product_id == "p-tool"


def test_search_latency_under_budget_at_scale():
    # 100k SKUs, single query well under the 200ms budget.
    catalog = Catalog()
    catalog.add_category(Category("c", "Cat"))
    # A realistically varied vocabulary: part types x brands/models, so posting
    # lists resemble a real catalog rather than a handful of giant lists.
    parts = ["screen", "battery", "cable", "camera", "speaker", "button", "frame",
             "glass", "chip", "flex", "connector", "digitizer", "backlight", "lens",
             "housing", "gasket", "adhesive", "bracket", "shield", "antenna"]
    brands = [f"brand{n}" for n in range(60)]
    models = [f"model{n}" for n in range(80)]
    rng = random.Random(42)
    for i in range(100_000):
        title = (
            f"{rng.choice(parts)} {rng.choice(parts)} {rng.choice(brands)} "
            f"{rng.choice(models)} sku {i}"
        )
        p = Product(
            id=f"p{i}",
            title=title,
            category="c",
            attributes={"brand": rng.choice(["A", "B", "C"])},
            popularity=rng.randint(0, 500),
        )
        v = Variant(sku=f"S{i}")
        v.set_price(PriceTier.RETAIL, Money(1000, "USD"))
        p.variants.append(v)
        catalog.add_product(p)
    index = SearchIndex(catalog, facet_attributes=["brand"])
    results = index.search("camera flex cable", limit=24)
    assert results.hits
    assert results.took_ms < 200, f"search took {results.took_ms}ms (budget 200ms)"
