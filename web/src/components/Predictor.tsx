import { useEffect, useMemo, useState } from 'react'
import { Section } from './Section'
import { Gauge } from './charts/Gauge'
import { ContributionBars } from './charts/ContributionBars'
import { Field, Select } from './PredictorFields'
import baseline from '../data/baseline.json'
import { explain, loadModel, predict, type Model, type Row } from '../lib/lgbm'
import { percentileOf } from '../lib/threshold'
import { prettyLabel } from '../lib/ui'
import {
  BASE_ROW, DAYS, EDUCATION, JOBS, MARITAL, MONTHS, PRESETS, REGIMES,
} from '../lib/profiles'

const DECISION = 0.5

export function Predictor() {
  const [model, setModel] = useState<Model | null>(null)
  const [failed, setFailed] = useState(false)
  const [row, setRow] = useState<Row>(BASE_ROW)
  const [regime, setRegime] = useState<string>(REGIMES[1].id)

  useEffect(() => {
    loadModel().then(setModel).catch(() => setFailed(true))
  }, [])

  const set = (patch: Row) => setRow((prev) => ({ ...prev, ...patch }))

  /** pdays and poutcome are structurally tied: no prior contact means no prior outcome. */
  const setContactedBefore = (yes: boolean) =>
    set(
      yes
        ? { was_contacted_before: 'yes', poutcome: 'failure', previous: Math.max(1, Number(row.previous)) }
        : { was_contacted_before: 'no', poutcome: 'nonexistent', previous: 0 },
    )

  const applyRegime = (id: string) => {
    const found = REGIMES.find((r) => r.id === id)
    if (!found) return
    setRegime(id)
    set(found.values)
  }

  const probability = useMemo(() => (model ? predict(model, row) : 0), [model, row])
  const drivers = useMemo(
    () => (model ? explain(model, row, baseline as Row) : []),
    [model, row],
  )

  const contactedBefore = row.was_contacted_before === 'yes'
  const percentile = percentileOf(probability)
  const call = probability >= DECISION

  return (
    <Section
      id="predictor"
      eyebrow="Try it"
      title="The model is running on this page. No server, no cold start."
      lede={
        <>
          The trained LightGBM booster — 250 trees, 46 encoded features — was flattened into a
          240&nbsp;KB JSON file and is being traversed in JavaScript as you change these fields. The
          export script asserts the browser reproduces{' '}
          <span className="font-mono text-ink">predict_proba</span> to within 1e-6, so this is the
          same number the Python pipeline returns.
        </>
      }
    >
      <div className="mb-4 grid gap-2 sm:grid-cols-3">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => {
              setRow(preset.row)
              const match = REGIMES.find(
                (r) => r.values['nr.employed'] === preset.row['nr.employed'],
              )
              if (match) setRegime(match.id)
            }}
            className="rounded-lg border border-white/10 bg-panel px-3.5 py-2 text-left transition-colors hover:border-white/25"
          >
            <span className="block text-[0.8125rem] font-medium text-ink">{preset.label}</span>
            <span className="mt-0.5 block max-w-[22rem] text-[0.75rem] leading-snug text-ink-3">
              {preset.blurb}
            </span>
          </button>
        ))}
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-[1.25fr_1fr]">
        <form className="panel p-5 sm:p-6" onSubmit={(e) => e.preventDefault()}>
          <fieldset className="m-0 border-0 p-0">
            <legend className="eyebrow mb-4">Customer</legend>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Field label="Age" htmlFor="age">
                <div className="flex items-center gap-3">
                  <input
                    id="age"
                    type="range"
                    min={18}
                    max={95}
                    value={Number(row.age)}
                    onChange={(e) => set({ age: Number(e.target.value) })}
                  />
                  <span className="num w-6 shrink-0 text-[0.875rem] text-ink tabular-nums">
                    {row.age}
                  </span>
                </div>
              </Field>
              <Select label="Occupation" id="job" value={String(row.job)} options={JOBS} onChange={(job) => set({ job })} />
              <Select label="Marital status" id="marital" value={String(row.marital)} options={MARITAL} onChange={(marital) => set({ marital })} />
              <Select label="Education" id="education" value={String(row.education)} options={EDUCATION} onChange={(education) => set({ education })} />
              <Select label="Housing loan" id="housing" value={String(row.housing)} options={['yes', 'no']} onChange={(housing) => set({ housing })} />
              <Select label="Personal loan" id="loan" value={String(row.loan)} options={['yes', 'no']} onChange={(loan) => set({ loan })} />
            </div>
          </fieldset>

          <fieldset className="m-0 mt-7 border-0 p-0">
            <legend className="eyebrow mb-4">This campaign</legend>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Select label="Channel" id="contact" value={String(row.contact)} options={['cellular', 'telephone']} onChange={(contact) => set({ contact })} />
              <Select label="Month" id="month" value={String(row.month)} options={MONTHS} onChange={(month) => set({ month })} />
              <Select label="Day" id="day" value={String(row.day_of_week)} options={DAYS} onChange={(day_of_week) => set({ day_of_week })} />
              <Field label="Calls so far" htmlFor="campaign">
                <div className="flex items-center gap-3">
                  <input
                    id="campaign"
                    type="range"
                    min={1}
                    max={20}
                    value={Number(row.campaign)}
                    onChange={(e) => set({ campaign: Number(e.target.value) })}
                  />
                  <span className="num w-6 shrink-0 text-[0.875rem] text-ink tabular-nums">
                    {row.campaign}
                  </span>
                </div>
              </Field>
              <Select
                label="Contacted before"
                id="before"
                value={contactedBefore ? 'yes' : 'no'}
                options={['yes', 'no']}
                onChange={(v) => setContactedBefore(v === 'yes')}
              />
              <Select
                label="Previous outcome"
                id="poutcome"
                value={String(row.poutcome)}
                options={contactedBefore ? ['failure', 'success'] : ['nonexistent']}
                disabled={!contactedBefore}
                onChange={(poutcome) => set({ poutcome })}
              />
            </div>
          </fieldset>

          <fieldset className="m-0 mt-7 border-0 p-0">
            <legend className="eyebrow mb-1.5">Market conditions</legend>
            <p className="mb-3 text-[0.75rem] leading-relaxed text-ink-3">
              Employment level, Euribor and consumer confidence are quarterly readings that move
              together, so they are set as historical regimes rather than free inputs — an arbitrary
              combination would be a customer the model never saw.
            </p>
            <div className="flex flex-wrap gap-2">
              {REGIMES.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  className="chip"
                  aria-pressed={regime === r.id}
                  onClick={() => applyRegime(r.id)}
                >
                  {r.label} · {r.period}
                </button>
              ))}
            </div>
            <p className="mt-3 text-[0.75rem] text-ink-3">
              {REGIMES.find((r) => r.id === regime)?.note} Actual conversion in this period:{' '}
              <span className="num text-ink-2 tabular-nums">
                {REGIMES.find((r) => r.id === regime)?.historical}%
              </span>
              .
            </p>
          </fieldset>
        </form>

        <div className="flex flex-col gap-4">
          <div className="panel flex flex-col items-center p-5 sm:p-6">
            {failed ? (
              <p role="alert" className="py-10 text-center text-[0.875rem] text-critical">
                The model file could not be loaded. Reload the page to try again.
              </p>
            ) : !model ? (
              <div
                role="status"
                aria-label="Loading the model"
                className="flex h-[13rem] w-full animate-pulse flex-col items-center justify-center gap-3"
              >
                <span className="h-24 w-56 rounded-t-full bg-white/5" />
                <span className="text-[0.75rem] text-ink-3">Loading 250 trees…</span>
              </div>
            ) : (
              <>
                <Gauge value={probability} threshold={DECISION} max={0.98} label="Subscription probability" />
                <div className="mt-3 flex items-center gap-2">
                  <span
                    aria-hidden
                    className="grid h-5 w-5 place-items-center rounded-full text-[0.75rem] font-bold"
                    style={{
                      background: call ? 'var(--color-series-3)' : 'rgba(255,255,255,0.09)',
                      color: call ? '#fff' : 'var(--color-ink-3)',
                    }}
                  >
                    {call ? '✓' : '–'}
                  </span>
                  <span className="text-[0.9375rem] font-medium text-ink">
                    {call ? 'Put on the call list' : 'Skip at a 50% cut-off'}
                  </span>
                </div>
                <p className="mt-2 text-center text-[0.8125rem] leading-relaxed text-ink-3">
                  Ranks above{' '}
                  <span className="num text-ink-2 tabular-nums">
                    {(percentile * 100).toFixed(1)}%
                  </span>{' '}
                  of the held-out book · {(probability / 0.1128).toFixed(1)}× the 11.3% portfolio
                  baseline
                </p>
              </>
            )}
          </div>

          <figure className="panel m-0 p-5 sm:p-6">
            <figcaption className="mb-4">
              <h3 className="text-[0.9375rem]">What moved the score</h3>
              <p className="mt-1 text-[0.8125rem] leading-relaxed text-ink-3">
                Each bar swaps one field back to the population baseline ({prettyLabel(String(baseline.job))},
                married, age {baseline.age}, first call) and re-scores. Blue lifts the probability,
                red drags it, in percentage points.
              </p>
            </figcaption>
            {model && <ContributionBars items={drivers} />}
            <p className="mt-4 text-[0.75rem] leading-relaxed text-ink-3">
              This is a leave-one-out attribution, not TreeSHAP — it is exact for the swap it
              describes, and unlike SHAP it needs no background dataset shipped to the browser.
              Effects interact, so the bars will not sum to the total.
            </p>
          </figure>
        </div>
      </div>
    </Section>
  )
}
