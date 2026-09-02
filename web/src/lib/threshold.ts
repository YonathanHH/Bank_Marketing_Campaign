import metrics from '../data/metrics.json'

const SCALE = 1000
const scores = metrics.scores as number[]
const labels = metrics.labels as number[]

export const TOTAL = scores.length
export const POSITIVES = labels.reduce((a, b) => a + b, 0)
export const NEGATIVES = TOTAL - POSITIVES

/**
 * Suffix histograms over the quantised scores: `posAtLeast[t]` is the number of
 * subscribers scored at or above t/1000. Every threshold query is then O(1)
 * instead of a scan over 8,040 rows on each slider frame.
 */
const posAtLeast = new Int32Array(SCALE + 2)
const negAtLeast = new Int32Array(SCALE + 2)

for (let i = 0; i < TOTAL; i++) {
  if (labels[i]) posAtLeast[scores[i]]++
  else negAtLeast[scores[i]]++
}
for (let t = SCALE - 1; t >= 0; t--) {
  posAtLeast[t] += posAtLeast[t + 1]
  negAtLeast[t] += negAtLeast[t + 1]
}

export type Outcome = {
  tp: number
  fp: number
  fn: number
  tn: number
  called: number
  precision: number
  recall: number
  f2: number
}

/** Confusion counts and rates for a probability cut-off in [0, 1]. */
export function outcomeAt(threshold: number): Outcome {
  const t = Math.min(SCALE, Math.max(0, Math.round(threshold * SCALE)))
  const tp = posAtLeast[t]
  const fp = negAtLeast[t]
  const fn = POSITIVES - tp
  const tn = NEGATIVES - fp
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp)
  const recall = tp / POSITIVES
  const f2 =
    precision + recall === 0 ? 0 : (5 * precision * recall) / (4 * precision + recall)
  return { tp, fp, fn, tn, called: tp + fp, precision, recall, f2 }
}

/** Cumulative gains: share of the list called → share of subscribers reached. */
export const gainsCurve: [number, number][] = (() => {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a])
  const points: [number, number][] = [[0, 0]]
  let found = 0
  for (let i = 0; i < TOTAL; i++) {
    found += labels[order[i]]
    if (i % Math.ceil(TOTAL / 120) === 0 || i === TOTAL - 1) {
      points.push([(i + 1) / TOTAL, found / POSITIVES])
    }
  }
  return points
})()

/** Share of the held-out book this probability outranks, as a percentile. */
export function percentileOf(probability: number): number {
  const t = Math.min(SCALE, Math.max(0, Math.round(probability * SCALE)))
  const above = posAtLeast[t] + negAtLeast[t]
  return 1 - above / TOTAL
}

export type Economics = { costPerCall: number; valuePerSale: number }

export function valueOf(outcome: Outcome, { costPerCall, valuePerSale }: Economics) {
  return outcome.tp * valuePerSale - outcome.called * costPerCall
}

/** The cut-off that maximises campaign value under the given unit economics. */
export function bestThreshold(economics: Economics): number {
  let best = 0
  let bestValue = -Infinity
  for (let t = 0; t <= SCALE; t += 5) {
    const value = valueOf(outcomeAt(t / SCALE), economics)
    if (value > bestValue) {
      bestValue = value
      best = t / SCALE
    }
  }
  return best
}
