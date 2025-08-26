# scripts/search_stac_element84.py - Search Element84 Earth Search
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import pystac_client

def load_cfg(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def aoi_union(aoi_dir: Path) -> gpd.GeoDataFrame:
    """Load and union all AOI geometries"""
    files = list(aoi_dir.glob("*.geojson"))
    if not files:
        raise ValueError(f"No AOI files in {aoi_dir}")
    
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g = g[g.geometry.notna()]
        if not g.empty:
            gdfs.append(g.to_crs("EPSG:4326"))
    
    if not gdfs:
        raise ValueError("AOI files contain no valid geometries.")
    
    import pandas as pd
    merged = pd.concat(gdfs, ignore_index=True).explode(index_parts=False, ignore_index=True)
    
    try:
        u = merged.geometry.union_all()
    except AttributeError:
        u = merged.geometry.unary_union
    
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def search_element84(aoi_geom, start_date, end_date, cloud_cover=20):
    """Search Element84 Earth Search for Sentinel-2 data"""
    
    # Connect to Element84 Earth Search
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    
    print(f"Searching Element84 from {start_date} to {end_date}")
    print(f"AOI bounds: {aoi_geom.total_bounds}")
    
    # Search parameters
    search = catalog.search(
        collections=["sentinel-2-l2a"],  # Element84 collection name
        intersects=aoi_geom.geometry.iloc[0].__geo_interface__,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {"lt": cloud_cover}
        },
        limit=1000  # Increase limit if needed
    )
    
    items = list(search.items())
    print(f"Found {len(items)} Sentinel-2 items")
    
    if not items:
        print("No items found. Try:")
        print("- Expanding date range")
        print("- Increasing cloud cover threshold")
        print("- Checking AOI coordinates")
        return []
    
    # Convert to the format your existing code expects
    formatted_items = []
    for item in items:
        formatted_item = {
            "id": item.id,
            "collection": item.collection_id,
            "item": item.to_dict()
        }
        formatted_items.append(formatted_item)
    
    # Print sample info
    if items:
        sample = items[0]
        print(f"Sample item: {sample.id}")
        print(f"Available assets: {list(sample.assets.keys())}")
        print(f"Cloud cover: {sample.properties.get('eo:cloud_cover', 'N/A')}%")
    
    return formatted_items

def main():
    parser = argparse.ArgumentParser(description="Search Element84 Earth Search for Sentinel-2")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--cloud_cover", type=int, default=20, help="Max cloud cover %")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_cfg(args.config)
    
    # Load AOI
    aoi = aoi_union(Path(cfg["paths"]["aoi_dir"]))
    print(f"AOI loaded with {len(aoi)} geometries")
    
    # Search
    try:
        items = search_element84(aoi, args.start, args.end, args.cloud_cover)
        
        if items:
            # Save results
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(items, f, indent=2)
            
            print(f"Saved {len(items)} items to {output_path}")
        else:
            print("No items found!")
            
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()