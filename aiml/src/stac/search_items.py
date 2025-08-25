import argparse, json, pathlib, yaml
from pystac_client import Client
import planetary_computer as pc

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(config, out, start, end, collections):
    cfg = load_cfg(config)
    cat = Client.open(cfg["stac"]["url"])
    aoi_dir = pathlib.Path(cfg["paths"]["aoi_dir"])
    geoms = []
    for g in aoi_dir.glob("mumbai_bbox.geojson"):
        geoms.append(json.loads(g.read_text(encoding="utf-8"))["features"][0]["geometry"])
    if not geoms:
        raise SystemExit(f"No AOIs found in {aoi_dir}")

    def search(collection, geom, start, end, cloud_lt=None, limit=100):
        q = dict(collections=[collection], datetime=f"{start}/{end}", intersects=geom, limit=limit)
        if "sentinel-2" in collection and cloud_lt is not None:
            q["query"] = {"eo:cloud_cover": {"lt": cloud_lt}}
        items = list(cat.search(**q).items())
        return [pc.sign(it).to_dict() for it in items]

    items_all = []
    for geom in geoms:
        for coll in collections:
            itms = search(coll, geom, start, end, cfg["stac"]["cloud_lt"])
            items_all.extend([{"aoi": "aoi", "collection": coll, "start": start, "end": end, "item": it} for it in itms])

    pathlib.Path(out).write_text(json.dumps(items_all, indent=2), encoding="utf-8")
    print(f"Wrote {len(items_all)} items -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", required=True)   # e.g. 2025-06-01
    ap.add_argument("--end", required=True)     # e.g. 2025-08-23
    ap.add_argument("--collections", nargs="+", default=["sentinel-2-l2a","sentinel-1-grd"])
    args = ap.parse_args()
    main(args.config, args.out, args.start, args.end, args.collections)
