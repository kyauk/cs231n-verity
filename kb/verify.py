"""End-to-end verification: ingest sample AV failure logs, search, cluster."""

from __future__ import annotations

from core.ingest import ingest_text
from core.search import search
from core.cluster import cluster_chunks, get_cluster

SAMPLE_LOGS = {
    "nhtsa:2024-001": (
        "On 03/15/2024 at approximately 14:32, a Waymo Jaguar I-PACE operating in "
        "autonomous mode on El Camino Real, Mountain View, CA struck a pedestrian who "
        "stepped off the curb between parked vehicles. The AV was traveling at 25 mph. "
        "The perception system detected the pedestrian 0.8 seconds before impact but "
        "the planning module did not initiate emergency braking in time. The pedestrian "
        "sustained minor injuries. Weather was clear and dry. The AV's lidar and camera "
        "feeds were reviewed; the lidar point cloud showed the pedestrian partially "
        "occluded by a parked SUV until 1.2 seconds before impact."
    ),
    "nhtsa:2024-002": (
        "On 04/22/2024, a Cruise Origin autonomous shuttle rear-ended a manually driven "
        "Honda Civic at a red light on Market Street, San Francisco. The shuttle was "
        "decelerating but the braking profile was insufficient due to wet road conditions. "
        "The radar-based adaptive cruise control underestimated stopping distance on the "
        "wet surface. No injuries reported. Post-incident analysis showed the friction "
        "coefficient used by the planner was calibrated for dry asphalt."
    ),
    "nhtsa:2024-003": (
        "Battery thermal runaway event in an autonomous delivery robot operating in "
        "Phoenix, AZ during a 115°F ambient temperature day. The robot's battery "
        "management system failed to throttle motor power when cell temperatures "
        "exceeded 60°C. The robot shut down on the sidewalk and emitted smoke. No fire "
        "occurred. Root cause traced to a firmware bug that ignored the upper thermal "
        "threshold after a recent OTA update."
    ),
    "nhtsa:2024-004": (
        "An autonomous truck operating on I-10 near Tucson, AZ experienced a sudden "
        "loss of GPS signal when passing under a highway overpass. The inertial "
        "navigation system (INS) accumulated drift of 2.3 meters over 8 seconds of GPS "
        "denial. The truck briefly crossed the lane boundary before GPS reacquired. "
        "The HD map matching module did not compensate for the lateral drift. No "
        "collision occurred but a near-miss was recorded with an adjacent vehicle."
    ),
    "nhtsa:2024-005": (
        "A Nuro R3 autonomous delivery vehicle failed to yield to an emergency vehicle "
        "with active sirens and lights on a residential street in Houston, TX. The "
        "vehicle's audio classification model misidentified the siren as construction "
        "noise. The vehicle continued at 15 mph for approximately 200 meters before a "
        "remote operator intervened. The emergency vehicle had to swerve around the Nuro "
        "unit. Investigation revealed the siren classifier had not been trained on the "
        "specific Federal Q siren type used by Houston Fire Department."
    ),
    "field:sensor-drift-001": (
        "IMU sensor drift increased significantly in humid conditions during monsoon "
        "season testing in Mumbai. The MEMS gyroscope showed a bias drift of 0.15 deg/s "
        "compared to the nominal 0.02 deg/s, causing cumulative navigation error of "
        "4.7 meters over a 10-minute autonomous run. The accelerometer was unaffected. "
        "Humidity inside the sensor enclosure reached 85% RH due to a degraded gasket "
        "seal. Replacing the gasket and adding a desiccant pack resolved the issue."
    ),
}


def main() -> None:
    # --- Step 1: Ingest ---
    print("=" * 60)
    print("STEP 1 — Ingesting sample AV failure logs")
    print("=" * 60)
    total = 0
    for source_id, text in SAMPLE_LOGS.items():
        n = ingest_text(source_id, text)
        print(f"  {source_id}: {n} chunk(s) inserted")
        total += n
    print(f"\n  Total chunks inserted: {total}\n")

    # --- Step 2: Search ---
    print("=" * 60)
    print("STEP 2 — Similarity search: 'battery overheating'")
    print("=" * 60)
    results = search("battery overheating", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n  #{i}  (similarity={r['similarity']:.4f})")
        print(f"      source: {r['source_id']}")
        print(f"      text:   {r['chunk_text'][:120]}...")

    print()
    print("=" * 60)
    print("STEP 2b — Similarity search: 'GPS signal loss navigation'")
    print("=" * 60)
    results2 = search("GPS signal loss navigation", top_k=3)
    for i, r in enumerate(results2, 1):
        print(f"\n  #{i}  (similarity={r['similarity']:.4f})")
        print(f"      source: {r['source_id']}")
        print(f"      text:   {r['chunk_text'][:120]}...")

    # --- Step 3: Cluster ---
    print()
    print("=" * 60)
    print("STEP 3 — Clustering into 3 groups")
    print("=" * 60)
    stats = cluster_chunks(n_clusters=3)
    print(f"\n  Silhouette score: {stats['silhouette']}")
    for c in stats["clusters"]:
        print(f"\n  Cluster {c['cluster_id']} ({c['count']} chunks):")
        print(f"    Representative: {c['representative'][:120]}...")

    # --- Step 4: Fetch one cluster ---
    print()
    print("=" * 60)
    print("STEP 4 — Fetching all chunks in cluster 0")
    print("=" * 60)
    members = get_cluster(0)
    for m in members:
        print(f"  id={m['id']}  source={m['source_id']}")
        print(f"    {m['chunk_text'][:100]}...")
        print()

    print("Done! Vector DB is functional.")


if __name__ == "__main__":
    main()
