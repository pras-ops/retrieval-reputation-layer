"""
Gate W: choosing a web-scraping tool per site, with real extraction code and native feedback.

The task RRL was first meant for. An agent must extract a product's title, price and SKU from
a shop's product page, and picks one extractor from a registry. Everything runs offline:

  * 30 generated shops on three page layouts: Shopify-like (theme v1, or v2 after a redesign),
    WooCommerce-like, and custom. Some pages embed schema.org JSON-LD, some microdata, most
    Open Graph meta tags.
  * 8 extractors, each real parsing code with a registry description: CSS extractors for each
    layout, JSON-LD, microdata, Open Graph (no SKU), a generic heuristic scraper, and a
    "universal AI extractor" whose description promises everything and whose output is guessed.
  * Feedback is produced by the tools, not simulated: the extractor's own success report
    (s_behave; the universal extractor always reports success), a schema validator (s_judge;
    well-formed guesses pass it), and a ground-truth spot check (s_gt) on a fraction of calls.
  * Redesign: in the `redesign` scenarios half the Shopify-like shops switch to theme v2
    halfway through, and some drop their JSON-LD.
  * 5% of calls fail on the network. 6 shops are held out until a final block.

The verifier is ground truth from the page generator, which is what a scraping team would get
from manually checked samples.

    python3 sim/run_gate_w.py --scenario all
    python3 sim/run_gate_w.py --selftest
"""

import argparse
import json
import os
import random
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.feedback import OutcomeSignals
from run_gate_matrix import (
    NO_DOC,
    RESULTS_DIR,
    enable_ablation,
    plot,
    print_report,
    run_policies,
    set_semantic,
    summarize,
)

N_SITES, N_HELDOUT, PAGES_PER_SITE = 30, 6, 4
NET_FAIL = 0.05
SKU_RE = re.compile(r"^[A-Z]{2,5}-\d{3,5}$")

SCENARIOS: Dict[str, dict] = {
    "stationary": {"coverage": 1.0, "switch": False},
    "spot_check": {"coverage": 0.2, "switch": False},
    "redesign": {"coverage": 1.0, "switch": True},
    "redesign_spot_check": {"coverage": 0.2, "switch": True},
}

# ---------------------------------------------------------------- sites and pages

WORDS = ["Trail", "Nova", "Aero", "Summit", "Urban", "Pixel", "Harbor", "Lumen", "Cedar", "Volt"]
ITEMS = ["Shoe", "Jacket", "Lamp", "Mug", "Backpack", "Watch", "Speaker", "Chair"]


def sites() -> List[dict]:
    rng = random.Random(2026)
    out = []
    for i in range(N_SITES):
        layout = rng.choices(["shopify", "woo", "custom"], weights=[4, 3, 3])[0]
        name = f"{rng.choice(WORDS).lower()}{rng.choice(ITEMS).lower()}{i}"
        out.append(
            {
                "id": i,
                "domain": f"{name}.shop",
                "prefix": name[:3].upper(),
                "layout": layout,
                "jsonld": rng.random() < 0.5,
                "microdata": rng.random() < 0.3,
                "og": rng.random() < 0.7,
                "hint": rng.random() < 0.5,  # whether the request names the platform
                "redesigns": layout == "shopify" and rng.random() < 0.5,
                "drops_jsonld": rng.random() < 0.3,
                "heldout": i >= N_SITES - N_HELDOUT,
            }
        )
    return out


def products(site: dict) -> List[dict]:
    rng = random.Random(site["id"] * 101 + 7)
    out = []
    for p in range(PAGES_PER_SITE):
        out.append(
            {
                "title": f"{rng.choice(WORDS)} {rng.choice(ITEMS)} {rng.randint(2, 99)}",
                "price": f"{rng.randint(5, 400)}.{rng.choice(['00', '49', '99'])}",
                "sku": f"{site['prefix']}-{rng.randint(100, 99999)}",
                "page": p,
            }
        )
    return out


def render(site: dict, prod: dict, after_switch: bool) -> str:
    t, pr, sku = prod["title"], prod["price"], prod["sku"]
    head = [f"<title>{t} | {site['domain']}</title>"]
    if site["og"]:
        head.append(f'<meta property="og:title" content="{t}">')
        head.append(f'<meta property="product:price:amount" content="{pr}">')
    if site["jsonld"] and not (after_switch and site["drops_jsonld"]):
        ld = {"@context": "https://schema.org", "@type": "Product", "name": t, "sku": sku}
        ld["offers"] = {"@type": "Offer", "price": pr, "priceCurrency": "USD"}
        head.append(f'<script type="application/ld+json">{json.dumps(ld)}</script>')
    layout = site["layout"]
    if layout == "shopify" and after_switch and site["redesigns"]:
        layout = "shopify_v2"
    body = {
        "shopify": (
            f'<h1 class="product__title">{t}</h1><span class="price-item">${pr}</span>'
            f'<span class="sku">SKU: {sku}</span>'
        ),
        "shopify_v2": (
            f'<h1 class="pdp-title">{t}</h1><div class="pdp-price" data-amount="{pr}">${pr}</div>'
            f'<p class="pdp-sku">{sku}</p>'
        ),
        "woo": (
            f'<h1 class="product_title entry-title">{t}</h1><p class="price"><span class='
            f'"woocommerce-Price-amount amount">{pr}&nbsp;$</span></p><span class="sku_wrapper">'
            f'SKU: <span class="sku">{sku}</span></span>'
        ),
        "custom": (
            f'<div class="name">{t}</div><div class="cost">USD {pr}</div>'
            f"<small>Item code {sku}</small>"
        ),
    }[layout]
    if site["microdata"]:
        body += (
            f'<div itemscope itemtype="https://schema.org/Product"><span itemprop="name">{t}'
            f'</span><meta itemprop="price" content="{pr}"><meta itemprop="sku" content="{sku}">'
            "</div>"
        )
    return f"<html><head>{''.join(head)}</head><body>{body}</body></html>"


# ---------------------------------------------------------------- extractors (the tools)


def _m(pattern: str, html: str) -> Optional[str]:
    m = re.search(pattern, html, re.S)
    return m.group(1).strip() if m else None


def ex_css_shopify(h):
    return {
        "title": _m(r'class="product__title">(.*?)<', h),
        "price": _m(r'class="price-item">\$?([\d.]+)<', h),
        "sku": _m(r'class="sku">SKU: (.*?)<', h),
    }


def ex_css_shopify_v2(h):
    return {
        "title": _m(r'class="pdp-title">(.*?)<', h),
        "price": _m(r'class="pdp-price" data-amount="([\d.]+)"', h),
        "sku": _m(r'class="pdp-sku">(.*?)<', h),
    }


def ex_css_woo(h):
    return {
        "title": _m(r'class="product_title[^"]*">(.*?)<', h),
        "price": _m(r'woocommerce-Price-amount amount">([\d.]+)', h),
        "sku": _m(r'<span class="sku">(.*?)<', h),
    }


def ex_jsonld(h):
    raw = _m(r'<script type="application/ld\+json">(.*?)</script>', h)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except Exception:
        return {}
    return {
        "title": d.get("name"),
        "price": (d.get("offers") or {}).get("price"),
        "sku": d.get("sku"),
    }


def ex_microdata(h):
    return {
        "title": _m(r'itemprop="name">(.*?)<', h),
        "price": _m(r'itemprop="price" content="([\d.]+)"', h),
        "sku": _m(r'itemprop="sku" content="(.*?)"', h),
    }


def ex_og(h):
    return {
        "title": _m(r'property="og:title" content="(.*?)"', h),
        "price": _m(r'property="product:price:amount" content="([\d.]+)"', h),
        "sku": None,
    }


def ex_heuristic(h):
    return {
        "title": _m(r"<h1[^>]*>(.*?)</h1>", h),
        "price": _m(r"\$\s?(\d+\.\d{2})", h),
        "sku": _m(r"SKU:\s*([A-Z]{2,5}-\d{3,5})", h),
    }


def ex_universal(h):
    # Looks right and is well-formed; the SKU is a guess.
    title = _m(r"<title>(.*?) \|", h)
    domain = _m(r"\| (.*?)</title>", h) or "xxx"
    return {"title": title, "price": _m(r"(\d+\.\d{2})", h), "sku": f"{domain[:3].upper()}-1000"}


TOOLS: List[Tuple[str, Callable[[str], dict], str]] = [
    (
        "css_shopify",
        ex_css_shopify,
        "CSS extractor for Shopify store product pages: product__title, price-item and sku.",
    ),
    (
        "css_shopify_v2",
        ex_css_shopify_v2,
        "CSS extractor for the new Shopify store theme product pages: pdp-title, pdp-price, "
        "pdp-sku.",
    ),
    (
        "css_woo",
        ex_css_woo,
        "CSS extractor for WooCommerce store product pages: product_title, "
        "woocommerce-Price-amount and sku.",
    ),
    (
        "jsonld",
        ex_jsonld,
        "Structured-data extractor: reads the schema.org Product JSON-LD embedded in a product "
        "page (name, offers.price, sku).",
    ),
    (
        "microdata",
        ex_microdata,
        "Microdata extractor: reads schema.org itemprop name, price and sku attributes.",
    ),
    ("og_meta", ex_og, "Open Graph meta-tag extractor for product title and price."),
    (
        "heuristic",
        ex_heuristic,
        "Generic heuristic scraper: finds the main heading, the price and the SKU text on a "
        "product page.",
    ),
    (
        "universal_ai",
        ex_universal,
        "Universal AI product extractor: extract product title, price and SKU from product "
        "pages of any store — Shopify store, WooCommerce store or custom store — with 99% "
        "accuracy.",
    ),
]


def evaluate(tool: int, site: dict, prod: dict, after_switch: bool) -> Tuple[float, float, float]:
    """(correct, self-reported success, schema-valid) for one extraction."""
    out = TOOLS[tool][1](render(site, prod, after_switch)) or {}
    title, price, sku = out.get("title"), out.get("price"), out.get("sku")
    correct = float(title == prod["title"] and price == prod["price"] and sku == prod["sku"])
    # Extractors report success when they found something; Open Graph never has a SKU and
    # reports success anyway.
    self_ok = float(
        bool(title) and bool(price) and (sku is not None or TOOLS[tool][0] == "og_meta")
    )
    try:
        price_ok = price is not None and float(price) > 0
    except ValueError:
        price_ok = False
    schema_ok = float(bool(title) and price_ok and bool(sku) and bool(SKU_RE.match(str(sku))))
    return correct, self_ok, schema_ok


def request_text(site: dict) -> str:
    platform = {"shopify": " (Shopify store)", "woo": " (WooCommerce store)", "custom": ""}
    hint = platform[site["layout"]] if site["hint"] else ""
    return f"Extract the product title, price and SKU from {site['domain']} product pages{hint}."


# ---------------------------------------------------------------- pools and replay


def build_pools(offline_mock: bool = False):
    """Rank all 8 tools for each site's request: hybrid similarity + cross-encoder order."""
    from run_gate_matrix import PoolBuilder, load_models

    embedder, cross_encoder = load_models(offline_mock=offline_mock)
    queries = [{"task_id": s["id"], "text": request_text(s)} for s in sites()]
    corpus = {i: {"code": f"{name}: {desc}"} for i, (name, _, desc) in enumerate(TOOLS)}
    import run_gate_matrix as m

    old_k, m.POOL_K = m.POOL_K, len(TOOLS)
    try:
        pools, ce = PoolBuilder(embedder, cross_encoder).build(queries, corpus)
    finally:
        m.POOL_K = old_k
    return pools, ce


def cosines(pools):
    """Raw MiniLM cosine of each site's request to every tool (for the RATED and MemRL ports)."""
    from run_gate_matrix import cosine_map, load_models

    embedder, _ = load_models(offline_mock=False)
    queries = [{"task_id": s["id"], "text": request_text(s)} for s in sites()]
    corpus = {i: {"code": f"{name}: {desc}"} for i, (name, _, desc) in enumerate(TOOLS)}
    return cosine_map(queries, corpus, pools, embedder)


def outcome_table():
    """(site, page, tool, after_switch) -> (correct, self_ok, schema_ok), all precomputed."""
    table = {}
    for s in sites():
        for prod in products(s):
            for tool in range(len(TOOLS)):
                for after in (False, True):
                    table[(s["id"], prod["page"], tool, after)] = evaluate(tool, s, prod, after)
    return table


def run_seed(seed: int, epochs: int, pools, ce, table, scenario: dict):
    """Requests are (site, page); pools and learning are per site's request text."""
    rng = random.Random(seed)
    all_sites = sites()
    seen = [(s["id"], p) for s in all_sites if not s["heldout"] for p in range(PAGES_PER_SITE)]
    held = [(s["id"], p) for s in all_sites if s["heldout"] for p in range(PAGES_PER_SITE)]
    stream_pairs: List[Tuple[int, int]] = []
    for _ in range(epochs):
        order = list(seen)
        rng.shuffle(order)
        stream_pairs.extend(order)
    switch_at = len(stream_pairs) // 2 if scenario["switch"] else 10**9
    rng.shuffle(held)
    stream_pairs.extend(held)
    net = [rng.random() < NET_FAIL for _ in stream_pairs]
    cov = [rng.random() < scenario["coverage"] for _ in stream_pairs]
    stream = [site for site, _ in stream_pairs]

    def cell(step: int, doc) -> Tuple[float, float, float]:
        if net[step] or doc == NO_DOC:
            return 0.0, 0.0, 0.0
        site, page = stream_pairs[step]
        return table[(site, page, int(doc), step >= switch_at)]

    def outcome(t: int, doc, step: int) -> float:
        return cell(step, doc)[0]

    def signals_fn(step: int, t: int, doc, r: float) -> OutcomeSignals:
        correct, self_ok, schema_ok = cell(step, doc)
        return OutcomeSignals(
            s_gt=correct if cov[step] else None, s_behave=self_ok, s_judge=schema_ok
        )

    sc = dict(scenario, signals_fn=signals_fn, sycophancy=0.0, judge_acc=1.0)
    return run_policies(seed, stream, pools, ce, outcome, sc), len(held)


def selftest() -> None:
    print("SELF-TEST (offline): extractors, redesign, feedback and replay.")
    s = {
        "id": 0,
        "domain": "trailshoe0.shop",
        "prefix": "TRA",
        "layout": "shopify",
        "jsonld": True,
        "microdata": False,
        "og": True,
        "hint": True,
        "redesigns": True,
        "drops_jsonld": False,
        "heldout": False,
    }
    p = products(s)[0]
    names = [n for n, _, _ in TOOLS]
    assert evaluate(names.index("css_shopify"), s, p, False)[0] == 1.0
    assert evaluate(names.index("css_shopify"), s, p, True)[0] == 0.0
    assert evaluate(names.index("css_shopify_v2"), s, p, True)[0] == 1.0
    assert evaluate(names.index("jsonld"), s, p, True)[0] == 1.0
    assert evaluate(names.index("og_meta"), s, p, False)[0] == 0.0
    c, self_ok, schema_ok = evaluate(names.index("universal_ai"), s, p, False)
    assert (c, self_ok, schema_ok) == (0.0, 1.0, 1.0), "universal extractor should fool both"
    pools, ce = build_pools(offline_mock=True)
    table = outcome_table()
    a, _ = run_seed(42, 2, pools, ce, table, SCENARIOS["redesign_spot_check"])
    b, _ = run_seed(42, 2, pools, ce, table, SCENARIOS["redesign_spot_check"])
    assert a == b, "replay is not deterministic"
    print("SELF-TEST OK")


def describe(table) -> dict:
    """Which tools work where: success rate per tool, before and after the redesign."""
    rates = {}
    for i, (name, _, _) in enumerate(TOOLS):
        for after in (False, True):
            cells = [v[0] for k, v in table.items() if k[2] == i and k[3] == after]
            rates[f"{name}{' (after redesign)' if after else ''}"] = sum(cells) / len(cells)
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--scenario", default="all", choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.ablation:
        enable_ablation()

    pools, ce = build_pools()
    set_semantic(cosines(pools))
    table = outcome_table()
    rates = describe(table)
    print(
        "tool success rates over all pages:", json.dumps({k: round(v, 2) for k, v in rates.items()})
    )
    names = [n for n, _, _ in TOOLS]
    top = {}
    for s in sites():
        top[names[ce[s["id"]][0]]] = top.get(names[ce[s["id"]][0]], 0) + 1
    print("reranker's first choice per site:", top)

    seeds = list(range(42, 42 + args.seeds))
    scen = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for sc_name in scen:
        per_seed, n_tail = {}, 0
        for sd in seeds:
            per_seed[sd], n_tail = run_seed(sd, args.epochs, pools, ce, table, SCENARIOS[sc_name])
        summary = summarize(per_seed, seeds)
        title = (
            f"GATE W — web-scraping tool selection, scenario={sc_name}, {len(seeds)} seeds x "
            f"{args.epochs} epochs, {N_SITES} sites ({N_HELDOUT} held out), {len(TOOLS)} tools"
        )
        print_report(summary, title)
        from run_gate_t import heldout_summary

        summary["heldout"] = heldout_summary(per_seed, seeds, n_tail)
        stem = f"gate_w_{sc_name}"
        with open(os.path.join(RESULTS_DIR, stem + ".json"), "w") as f:
            json.dump(
                {
                    "title": title,
                    "scenario": SCENARIOS[sc_name],
                    "tool_success_rates": rates,
                    "reranker_first_choice": top,
                    "seeds": seeds,
                    "epochs": args.epochs,
                    **summary,
                },
                f,
                indent=2,
            )
        plot(per_seed, seeds, os.path.join(RESULTS_DIR, stem + ".png"), title=f"Gate W — {sc_name}")


if __name__ == "__main__":
    main()
