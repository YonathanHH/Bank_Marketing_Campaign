import { useMemo, useState } from 'react'
import { Section } from './Section'
import { ChartFrame } from './charts/ChartFrame'
import { ConfusionGrid } from './charts/ConfusionGrid'
import { LineChart } from './charts/LineChart'
import { eur, num } from '../lib/ui'
import {
  POSITIVES,
  TOTAL,
  bestThreshold,
  gainsCurve,
  outcomeAt,
  valueOf,
  type Economics,
} from '../lib/threshold'

const DEFAULTS: Economics = { costPerCall: 6, valuePerSale: 90 }

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <dt className="text-[0.6875rem] tracking-wide text-ink-3 uppercase">{label}</dt>
      <dd className="metric num m-0 mt-1.5 text-[1.375rem] tabular-nums">{value}</dd>
      {hint && <p className="mt-1 text-[0.6875rem] text-ink-3">{hint}</p>}
    </div>
  )
}

function MoneyInput({
  id,
  label,
  value,
  onChange,
}: {
  id: string
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-[0.75rem] text-ink-3">
        {label}
      </label>
      <div className="mt-1.5 flex items-center gap-1.5">
        <span aria-hidden className="text-[0.875rem] text-ink-3">
          €
        </span>
        <input
          id={id}
          type="number"
          min={0}
          max={100000}
          value={value}
          onChange={(e) => onChange(Math.max(0, Number(e.target.value) || 0))}
          className="field num tabular-nums"
        />
      </div>
    </div>
  )
}

export function ThresholdLab() {
  const [threshold, setThreshold] = useState(0.5)
  const [economics, setEconomics] = useState(DEFAULTS)

  const outcome = useMemo(() => outcomeAt(threshold), [threshold])
  const optimal = useMemo(() => bestThreshold(economics), [economics])

  const value = valueOf(outcome, economics)
  const callAll = valueOf(outcomeAt(0), economics)
  const share = outcome.called / TOTAL

  return (
    <Section
      id="threshold"
      eyebrow="Threshold lab"
      title="The model outputs a probability. Someone still has to draw the line."
      lede="Every cut-off buys recall with wasted calls. Drag the threshold and watch the same 8,040 held-out customers get re-sorted into money won and money burned — then let the unit economics pick the line for you."
    >
      <div className="grid items-start gap-4 lg:grid-cols-[1.05fr_1fr]">
        <div className="panel p-5 sm:p-6">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <label htmlFor="threshold-input" className="text-[0.9375rem] font-semibold text-ink">
              Call everyone scoring above
            </label>
            <output
              htmlFor="threshold-input"
              className="metric num text-[1.75rem] tabular-nums"
              style={{ color: 'var(--color-series-1)' }}
            >
              {(threshold * 100).toFixed(0)}%
            </output>
          </div>
          <input
            id="threshold-input"
            type="range"
            min={1}
            max={95}
            step={1}
            value={Math.round(threshold * 100)}
            onChange={(e) => setThreshold(Number(e.target.value) / 100)}
            className="mt-3"
            style={{
              // filled portion of the track carries the current cut-off
              ['--track' as string]: `linear-gradient(90deg, var(--color-series-1) ${(threshold - 0.01) / 0.94 * 100}%, var(--color-grid) 0)`,
            }}
            aria-describedby="threshold-summary"
          />
          <p id="threshold-summary" className="mt-1 text-[0.75rem] text-ink-3">
            {num(outcome.called)} of {num(TOTAL)} customers called ({(share * 100).toFixed(1)}% of
            the list) · {num(outcome.tp)} of {num(POSITIVES)} subscribers reached
          </p>

          <dl className="mt-6 grid grid-cols-3 gap-4 border-y border-white/8 py-5">
            <Stat
              label="Precision"
              value={`${(outcome.precision * 100).toFixed(1)}%`}
              hint="of calls convert"
            />
            <Stat
              label="Recall"
              value={`${(outcome.recall * 100).toFixed(1)}%`}
              hint="of subscribers found"
            />
            <Stat label="F2" value={outcome.f2.toFixed(3)} hint="recall-weighted" />
          </dl>

          <div className="mt-5">
            <ConfusionGrid counts={outcome} />
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="panel p-5 sm:p-6">
            <h3 className="text-[1.0625rem]">What the line is worth</h3>
            <p className="mt-2 text-[0.8125rem] leading-relaxed text-ink-3">
              The dataset carries no costs, so these are assumptions — change them and the
              recommended cut-off moves with them.
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <MoneyInput
                id="cost-per-call"
                label="Cost of one call"
                value={economics.costPerCall}
                onChange={(costPerCall) => setEconomics((e) => ({ ...e, costPerCall }))}
              />
              <MoneyInput
                id="value-per-sale"
                label="Margin per subscription"
                value={economics.valuePerSale}
                onChange={(valuePerSale) => setEconomics((e) => ({ ...e, valuePerSale }))}
              />
            </div>

            <dl className="mt-6 grid grid-cols-2 gap-5 border-t border-white/8 pt-5">
              <Stat
                label="Campaign value at this line"
                value={eur(value)}
                hint={`${eur(outcome.tp * economics.valuePerSale)} won − ${eur(outcome.called * economics.costPerCall)} spent`}
              />
              <Stat
                label="Versus calling everyone"
                value={`${value - callAll >= 0 ? '+' : '−'}${eur(Math.abs(value - callAll))}`}
                hint={`Blind dialling all ${num(TOTAL)} returns ${eur(callAll)}`}
              />
            </dl>

            <div className="mt-5 flex flex-wrap items-center gap-3 rounded-lg border border-white/8 bg-white/[0.03] p-4">
              <p className="m-0 flex-1 text-[0.8125rem] leading-relaxed text-ink-2">
                At €{economics.costPerCall} a call and €{economics.valuePerSale} a subscription, value
                peaks at a{' '}
                <span className="num text-ink tabular-nums">{(optimal * 100).toFixed(0)}%</span>{' '}
                cut-off, returning {eur(valueOf(outcomeAt(optimal), economics))}.
              </p>
              <button
                type="button"
                onClick={() => setThreshold(optimal)}
                className="shrink-0 rounded-md border border-white/15 px-3 py-1.5 text-[0.8125rem] text-ink transition-colors hover:border-white/30 hover:bg-white/5"
              >
                Jump to it
              </button>
            </div>
          </div>

          <ChartFrame
            title="Cumulative gains"
            subtitle="Work down the ranked list and this is how fast you find the subscribers."
            legend={[
              { label: 'Model ranking', color: 'var(--color-series-1)' },
              { label: 'Random order', color: 'var(--color-ink-3)', dashed: true },
            ]}
            note={`Calling the top-scoring 10% of the list reaches 47.3% of all subscribers; the top 20% reaches 66.3%. That is the whole business case in one curve — and it holds whatever threshold you settle on, because ranking is what the model is really good at.`}
          >
            <LineChart
              ariaLabel="Cumulative gains curve: the top 10 percent of the ranked list contains 47.3 percent of subscribers"
              series={[
                { label: 'Model ranking', color: 'var(--color-series-1)', points: gainsCurve },
                {
                  label: 'Random order',
                  color: 'var(--color-ink-3)',
                  points: [
                    [0, 0],
                    [1, 1],
                  ],
                  dashed: true,
                },
              ]}
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              xTicks={[0, 0.25, 0.5, 0.75, 1]}
              yTicks={[0, 0.25, 0.5, 0.75, 1]}
              xLabel="Share of the list called"
              yLabel="Share of subscribers reached"
              format={(v) => `${(v * 100).toFixed(0)}%`}
              height={260}
            />
          </ChartFrame>
        </div>
      </div>
    </Section>
  )
}
