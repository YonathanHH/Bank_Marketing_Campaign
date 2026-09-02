import { useEffect, useState } from 'react'
import metrics from '../data/metrics.json'
import eda from '../data/eda.json'

const BASELINE = Math.round(eda.rate)
const TOP_DECILE = Math.round(metrics.deciles[0].rate)
const CAPTURED = metrics.deciles[0].captured

type Mode = 'blind' | 'model'

/** Deterministic scatter so the highlighted cells never look like a diagonal. */
const ORDER = Array.from({ length: 100 }, (_, i) => (i * 37) % 100)

function DotGrid({ mode }: { mode: Mode }) {
  const lit = mode === 'blind' ? BASELINE : TOP_DECILE

  return (
    <div
      className="grid grid-cols-10 gap-[6px]"
      role="img"
      aria-label={
        mode === 'blind'
          ? `Grid of 100 dots, ${BASELINE} highlighted: ${BASELINE} in 100 blind calls end in a subscription`
          : `Grid of 100 dots, ${TOP_DECILE} highlighted: ${TOP_DECILE} in 100 calls end in a subscription when the model picks who to ring`
      }
    >
      {ORDER.map((slot, i) => {
        const on = slot < lit
        return (
          <span
            key={i}
            className="aspect-square rounded-[3px]"
            style={{
              background: on ? 'var(--color-series-3)' : 'rgba(255,255,255,0.07)',
              transition: 'background-color 420ms ease',
              transitionDelay: `${(slot % 25) * 14}ms`,
            }}
          />
        )
      })}
    </div>
  )
}

export function Hero() {
  const [mode, setMode] = useState<Mode>('blind')
  const [touched, setTouched] = useState(false)

  useEffect(() => {
    if (touched) return
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    const timer = setTimeout(() => setMode('model'), 2200)
    return () => clearTimeout(timer)
  }, [touched])

  const pick = (next: Mode) => {
    setTouched(true)
    setMode(next)
  }

  return (
    <section id="top" className="relative overflow-hidden pt-28 pb-16 sm:pt-36 sm:pb-24">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 -top-40 h-[32rem] opacity-70"
        style={{
          background:
            'radial-gradient(48rem 22rem at 22% 0%, rgba(57,135,229,0.16), transparent 70%)',
        }}
      />
      <div className="shell relative grid gap-14 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:gap-20">
        <div>
          <p className="eyebrow">Yonathan Hary Hutagalung · Data Science Portfolio</p>
          <h1 className="mt-5 text-[2.5rem] leading-[1.05] font-semibold sm:text-[3.5rem]">
            Who answers <span className="text-series-3">yes</span>?
          </h1>
          <p className="mt-6 max-w-xl text-[1.0625rem] leading-relaxed text-ink-2">
            A Portuguese bank made 41,188 term-deposit sales calls and closed 11% of them. This is
            the propensity model that decides which calls are worth making — trained in Python,
            exported to JSON, and scoring{' '}
            <span className="text-ink">live in your browser on this page</span>.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="#predictor"
              className="rounded-lg bg-ink px-4 py-2.5 text-[0.875rem] font-medium text-plane no-underline transition-opacity hover:opacity-88"
            >
              Score a customer →
            </a>
            <a
              href="#leakage"
              className="rounded-lg border border-white/15 px-4 py-2.5 text-[0.875rem] font-medium text-ink-2 no-underline transition-colors hover:border-white/30 hover:text-ink"
            >
              The result I threw away
            </a>
          </div>

          <dl className="mt-12 grid grid-cols-2 gap-x-8 gap-y-6 border-t border-white/8 pt-8 sm:grid-cols-4">
            {[
              { v: '41,188', k: 'calls analysed' },
              { v: metrics.headline.rocAuc.toFixed(2), k: 'ROC-AUC, no leakage' },
              { v: `${CAPTURED}%`, k: 'of subscribers in the top decile' },
              { v: '0', k: 'servers in the loop' },
            ].map((stat) => (
              <div key={stat.k}>
                <dt className="sr-only">{stat.k}</dt>
                <dd className="m-0">
                  <span className="metric num block text-[1.75rem] tabular-nums">{stat.v}</span>
                  <span className="mt-1.5 block text-[0.75rem] leading-snug text-ink-3">
                    {stat.k}
                  </span>
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <figure className="panel m-0 p-5 sm:p-6">
          <figcaption className="mb-5">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-[0.9375rem] font-semibold">Out of every 100 calls</h2>
              <div className="flex gap-1.5" role="group" aria-label="Call list strategy">
                <button type="button" className="chip" aria-pressed={mode === 'blind'} onClick={() => pick('blind')}>
                  Blind
                </button>
                <button type="button" className="chip" aria-pressed={mode === 'model'} onClick={() => pick('model')}>
                  Model
                </button>
              </div>
            </div>
          </figcaption>

          <DotGrid mode={mode} />

          <div className="mt-5 flex items-baseline gap-3 border-t border-white/8 pt-4">
            <span
              className="metric num text-[2rem] tabular-nums"
              style={{ color: 'var(--color-series-3)' }}
            >
              {mode === 'blind' ? BASELINE : TOP_DECILE}
            </span>
            <p className="m-0 text-[0.8125rem] leading-snug text-ink-2">
              {mode === 'blind'
                ? 'subscriptions, dialling the list in the order it arrived.'
                : `subscriptions, dialling only the top decile of the model's ranking — ${(TOP_DECILE / BASELINE).toFixed(1)}× the hit rate for the same effort.`}
            </p>
          </div>
          <p className="mt-3 text-[0.6875rem] text-ink-3">
            Measured on the 8,040-call held-out test set, not the training data.
          </p>
        </figure>
      </div>
    </section>
  )
}
