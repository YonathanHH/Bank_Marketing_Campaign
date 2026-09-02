/**
 * Client-side scorer for the exported LightGBM pipeline.
 *
 * `tools/export_model.py` flattens the booster into typed arrays and asserts that
 * this traversal reproduces scikit-learn's `predict_proba` to within 1e-6, so the
 * numbers here are the same ones the Python model returns — no server round-trip.
 */

export type Row = Record<string, string | number>

type Tree = {
  root: number
  feature: number[]
  threshold: number[]
  left: number[]
  right: number[]
  leaf: number[]
}

export type Model = {
  numeric: { name: string; mean: number; scale: number }[]
  /** `base` is the level dropped by OneHotEncoder(drop='first'); `null` is the unknown level. */
  categorical: { name: string; base: string | null; levels: (string | null)[] }[]
  nFeatures: number
  trees: Tree[]
}

const UNKNOWN = 'unknown'

let cache: Promise<Model> | null = null

export function loadModel(): Promise<Model> {
  cache ??= fetch(`${import.meta.env.BASE_URL}model.json`).then((res) => {
    if (!res.ok) throw new Error(`model.json returned ${res.status}`)
    return res.json() as Promise<Model>
  })
  return cache
}

/** Build the 46-column design matrix row: scaled numerics, then one-hot levels. */
export function encode(model: Model, row: Row): Float64Array {
  const x = new Float64Array(model.nFeatures)
  let at = 0

  for (const spec of model.numeric) {
    x[at++] = (Number(row[spec.name]) - spec.mean) / spec.scale
  }

  for (const spec of model.categorical) {
    const raw = row[spec.name]
    const value = raw === undefined || raw === UNKNOWN ? null : String(raw)
    for (const level of spec.levels) {
      x[at++] = level === value ? 1 : 0
    }
  }
  return x
}

/** Sum of leaf values across every tree, i.e. the raw margin before the sigmoid. */
function margin(model: Model, x: Float64Array): number {
  let total = 0
  for (const tree of model.trees) {
    let node = tree.root
    while (node >= 0) {
      node = x[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node]
    }
    total += tree.leaf[-node - 1]
  }
  return total
}

export function predict(model: Model, row: Row): number {
  return 1 / (1 + Math.exp(-margin(model, encode(model, row))))
}

export type Contribution = { field: string; delta: number; value: string }

/**
 * Leave-one-out attribution: swap a single field back to the population baseline
 * and measure how far the probability moves. Cheap, exact for the model as given,
 * and honest about what it is — not TreeSHAP, which needs the training background.
 */
export function explain(model: Model, row: Row, baseline: Row): Contribution[] {
  const full = predict(model, row)
  const fields = [...model.numeric, ...model.categorical].map((f) => f.name)

  return fields
    .filter((field) => String(row[field]) !== String(baseline[field]))
    .map((field) => ({
      field,
      value: String(row[field]),
      delta: full - predict(model, { ...row, [field]: baseline[field] }),
    }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
}
