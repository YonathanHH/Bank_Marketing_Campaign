/**
 * Parity test: the browser scorer must agree with scikit-learn's predict_proba.
 *
 *   python tools/make_fixtures.py && node tools/check_parity.mjs
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const model = JSON.parse(readFileSync(join(HERE, '..', 'web', 'public', 'model.json'), 'utf8'))
const { cases } = JSON.parse(readFileSync(join(HERE, 'fixtures.json'), 'utf8'))

const UNKNOWN = 'unknown'

function encode(row) {
  const x = new Float64Array(model.nFeatures)
  let at = 0
  for (const spec of model.numeric) x[at++] = (Number(row[spec.name]) - spec.mean) / spec.scale
  for (const spec of model.categorical) {
    const raw = row[spec.name]
    const value = raw === undefined || raw === null || raw === UNKNOWN ? null : String(raw)
    for (const level of spec.levels) x[at++] = level === value ? 1 : 0
  }
  return x
}

function predict(row) {
  const x = encode(row)
  let total = 0
  for (const tree of model.trees) {
    let n = tree.root
    while (n >= 0) n = x[tree.feature[n]] <= tree.threshold[n] ? tree.left[n] : tree.right[n]
    total += tree.leaf[-n - 1]
  }
  return 1 / (1 + Math.exp(-total))
}

let worst = 0
let failures = 0
for (const [i, testCase] of cases.entries()) {
  const got = predict(testCase.row)
  const err = Math.abs(got - testCase.expected)
  worst = Math.max(worst, err)
  if (err > 1e-6) {
    failures++
    console.error(`case ${i}: expected ${testCase.expected.toFixed(8)}, got ${got.toFixed(8)}`)
  }
}

console.log(`${cases.length} fixtures · worst |JS − sklearn| = ${worst.toExponential(2)}`)
if (failures) {
  console.error(`FAIL — ${failures} case(s) outside 1e-6`)
  process.exit(1)
}
console.log('PASS — browser scorer matches scikit-learn')
