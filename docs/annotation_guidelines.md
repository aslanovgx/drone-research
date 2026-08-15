# Crop Annotation Guidelines — Classifier Stage

These rules govern how a single **image crop** produced by the SAM stage is
assigned exactly one label. They exist so that two different annotators labelling
the same crop reach the same answer.

## Scope

The SAM stage segments a raw drone image into candidate masks. Each mask is
converted to a bounding box and cropped out of the source image, producing files
named:

```
outputs/crops/segment_<id>.png
```

where `<id>` is the integer index of the mask within its source image (e.g.
`segment_17.png`). One crop = one training sample = one label.

The classifier is trained on **four** classes:

| Index | Class      |
|-------|------------|
| 0     | `building` |
| 1     | `car`      |
| 2     | `other`    |
| 3     | `tree`     |

The index order is alphabetical and is derived automatically from the class list
in `configs/classifier.yaml` — see `src/classification/dataset.py`. Never rely on
the order the classes appear in prose; rely on the sorted mapping.

> **`road` is intentionally excluded from this task.** SAM currently fragments
> road surfaces into many thin, inconsistent masks, so road crops are not stable
> enough to label. Road-like crops are handled by the fallback rules below until
> that issue is resolved and a `road` class is added in a later task.

## Labelling procedure

For every crop, in this order:

1. **Is the crop usable at all?** If it is background-only, near-empty, or too
   degraded to interpret → `other`. Stop.
2. **Is there one dominant object?** Estimate the fraction of crop area occupied
   by the single largest recognisable object. If that object covers **≥ 50%** of
   the crop and is identifiable, label the crop with that object's class. Stop.
3. **Otherwise** → `other`.

The 50% dominance threshold is the tie-breaker for every "there is a bit of
everything in here" case. Judge it by eye; do not measure precisely.

---

## Class definitions

### `building`

**Counts as an instance:** a man-made roofed structure viewed from above or at an
oblique drone angle — houses, warehouses, packing-house halls, sheds, silos,
covered loading docks, greenhouses with a solid roof.

Include when:
- The crop is dominated by a roof plane, with or without visible walls.
- Rooftop clutter (HVAC units, vents, skylights, solar panels, water tanks) is
  present — the crop is still `building`.
- Only part of a large building is captured, but roof texture/edges make it
  unambiguously a building (a corner, a roof ridge, a clean roof-edge shadow).
- A row of adjoining structures fills the crop — the class is per-crop, not
  per-instance, so a terrace of several roofs is still `building`.

Exclude (label otherwise):
- Paved yards, parking aprons and concrete slabs with no roof → `other`.
- Shipping containers, trailers detached from a cab, and open canopies/carports
  with no enclosed structure → `other`.
- A featureless flat colour patch that *might* be a roof but has no edge, shadow,
  or texture cue → `other`.

### `tree`

**Counts as an instance:** vegetation with visible canopy structure — individual
trees, tree clusters, orchard rows, large shrubs read from above as a crown.

Include when:
- One or several canopies dominate, including overlapping crowns where individual
  trees cannot be separated.
- The canopy is bare/leafless but branch structure is clearly a tree.
- The crop is a portion of a large canopy, provided leaf/branch texture is
  visible.

Exclude (label otherwise):
- Flat lawn, grass verges, hedgerows without crown structure, planted field rows,
  bare soil, and low ground cover → `other`. `tree` means canopy, not "green".
- A tree *shadow* cast on the ground, with no canopy in the crop → `other`.

### `car`

**Counts as an instance:** a road-going vehicle — cars, vans, pickups, SUVs,
lorries/trucks, buses, and motorcycles.

Include when:
- The vehicle is the dominant object, whether parked or in motion (motion blur is
  fine).
- The vehicle is partially cut off by the crop border but its shape (roof,
  windscreen, wheels, body outline) is still identifiable.
- Multiple vehicles in a row fill the crop — still `car`.

Exclude (label otherwise):
- Trailers, containers, and machinery detached from a vehicle body → `other`.
- An empty parking bay, painted bay markings, or a vehicle-shaped shadow with no
  vehicle → `other`.
- A blob whose vehicle identity rests only on being in a car park → `other`.

### `other`

`other` is the explicit catch-all and **must not** be treated as a dumping ground
for laziness — it has its own positive definition. Use it when the crop is any of:

- **Background-only:** ground, soil, gravel, water, sky, uniform shadow, or a
  featureless colour field.
- **Road / paved surface:** asphalt, lane markings, kerbs, pavements, parking
  aprons. Deliberately here, not in a `road` class, for this iteration.
- **Ambiguous fragment:** a sliver, edge, or fragment too small or too partial to
  identify — the typical SAM over-segmentation artefact.
- **Mixed content:** several classes present with no single object ≥ 50% of the
  crop area (e.g. half roof, half canopy).
- **Occlusion:** the object of interest is mostly hidden behind another object,
  deep shadow, or glare, so its class cannot be established.
- **Other real objects outside the three target classes:** solar farms, pools,
  fences, poles, street furniture, boats, livestock, people, agricultural
  machinery.
- **Degraded crops:** heavy blur, extreme over/under-exposure, compression
  artefacts, or crops smaller than ~16×16 px.

## Edge cases (decided once, applied everywhere)

| Situation | Label |
|---|---|
| Car under a tree canopy, car mostly visible | `car` |
| Car under a tree canopy, mostly covered by leaves | `other` (occlusion) |
| Rooftop with a parked vehicle on it, roof dominant | `building` |
| Building with a large tree in front, tree ≥ 50% | `tree` |
| Roof + canopy roughly 50/50, no clear dominance | `other` (mixed) |
| Long thin mask along a roof ridge, roof texture readable | `building` |
| Long thin mask along a road edge | `other` |
| Shadow of a building/tree/car only | `other` |
| Solar panels mounted on a roof, roof visible | `building` |
| Ground-mounted solar array | `other` |
| Greenhouse with translucent roof, structure readable | `building` |
| Orchard rows seen as repeating crowns | `tree` |
| Grass/lawn or hedge with no crown structure | `other` |
| Truck trailer without a cab | `other` |
| Crop is >90% empty border padding | `other` |

## Consistency rules

- **One label per crop.** No multi-label, no "unknown" bucket — `other` is the
  bucket.
- **Label what is in the crop**, not what you know is in the surrounding scene.
  Do not open the source image to disambiguate.
- **When genuinely undecided between a target class and `other`, choose `other`.**
  A false `building`/`tree`/`car` is more damaging than an extra `other`.
- **When undecided between two target classes**, apply the 50% dominance rule; if
  it still does not resolve, choose `other`.
- Record uncertain calls in the review queue rather than inventing a new rule;
  new rules land in this document so they apply retroactively to everyone.

## Dataset directory structure

Labelled crops are filed by split and class. This layout is exactly what
`src/classification/dataset.py` expects:

```
data/classifier/
├── train/
│   ├── building/
│   ├── tree/
│   ├── car/
│   └── other/
└── validation/
    ├── building/
    ├── tree/
    ├── car/
    └── other/
```

- Every class directory must exist in **both** splits, even if temporarily empty.
- Files keep their SAM-assigned name, `segment_<id>.png`. If ids collide across
  source images, prefix with the source scene id (`<scene>_segment_<id>.png`) —
  the loader only requires the `segment_<id>` fragment to be present for
  inference-time id parsing.
- Labelling assigns a crop to a directory; **no sidecar label files** are used.
- Split by **source image**, never by crop: all crops from one drone frame go to
  the same split, otherwise near-duplicate crops leak between train and
  validation.
- Suggested split ratio: 80% train / 20% validation, keeping every class present
  in both.
- `data/classifier/**` is gitignored. Crops, like all datasets, are never
  committed.

## Current status of the data

The real crops are not labelled yet — the SAM stage's output for the ArcGIS
Packing House District scenes still has to be reviewed. Until then,
`scripts/generate_sample_crops.py` writes **synthetic placeholder PNGs** into the
structure above so the dataset loader, training loop and inference path can be
exercised end to end. Those placeholders are not meaningful imagery and must be
deleted before real training.

Final crops will be labelled according to the rules in this document once the
SAM stage output is available.
