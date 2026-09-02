import { useState } from 'react'
import { Section } from './Section'
import { ChartFrame } from './charts/ChartFrame'
import { RateBars, type RateRow } from './charts/RateBars'
import eda from '../data/eda.json'

type Dim = {
  key: string
  label: string
  rows: RateRow[]
  raw?: boolean
  finding: string
}

/** Buckets thinner than this are noise on a rate chart, so they are not drawn. */
const MIN_N = 150

const DIMS: Dim[] = [
  {
    key: 'poutcome',
    label: 'Previous outcome',
    rows: eda.poutcome,
    finding:
      'Someone who already said yes to a previous campaign converts at 64.9% — nearly six times the baseline. Past behaviour is the single most useful thing on the record.',
  },
  {
    key: 'euribor',
    label: 'Interest rate',
    rows: eda.euribor.filter((r) => r.n >= MIN_N),
    raw: true,
    finding:
      'When the 3-month Euribor sat below 1%, 45.6% of calls converted. Above 4%, 4.9% did. A term deposit competes with every other place to park money, and cheap credit makes it look good.',
  },
  {
    key: 'month',
    label: 'Month',
    rows: eda.month,
    finding:
      'March, September, October and December convert around seven times better than May — but May carries a third of the call volume. The calendar effect is largely the rate cycle in disguise, which is why month on its own is a trap.',
  },
  {
    key: 'age',
    label: 'Age band',
    rows: eda.age,
    raw: true,
    finding:
      'Conversion is U-shaped: the over-65s convert at 46.8% and the under-25s at 20.9%, while the 36–55 working core sits below baseline. A linear age term would miss this entirely.',
  },
  {
    key: 'previousBand',
    label: 'Prior contacts',
    rows: eda.previousBand,
    raw: true,
    finding:
      'Contacts carried over from earlier campaigns convert at 21% after one prior touch and 59% after three. The catch: 86% of the book has never been contacted before, so this signal is absent exactly where the volume is.',
  },
  {
    key: 'job',
    label: 'Occupation',
    rows: eda.job,
    finding:
      'Students and retirees lead; blue-collar workers and entrepreneurs trail. This mostly restates the age curve rather than adding new signal.',
  },
  {
    key: 'campaignBand',
    label: 'Calls made',
    rows: eda.campaignBand.filter((r) => r.n >= MIN_N),
    raw: true,
    finding:
      'Conversion decays monotonically with every additional call in the same campaign — 13.0% on the first attempt, 3.2% past the tenth. Persistence is not a strategy; it is a cost.',
  },
  {
    key: 'education',
    label: 'Education',
    rows: eda.education.filter((r) => r.n >= MIN_N),
    finding:
      'A real but modest gradient: 13.8% for university graduates against 7.8% for basic schooling. Useful to the model, useless as a targeting rule on its own.',
  },
  {
    key: 'contact',
    label: 'Channel',
    rows: eda.contact,
    finding:
      'Mobile contacts convert at roughly triple the landline rate — partly a channel effect, partly that the landline records cluster in the early, high-rate campaigns.',
  },
]

export function DataSection() {
  const [active, setActive] = useState(DIMS[0])
  const share = eda.positives / eda.rows

  return (
    <Section
      id="data"
      eyebrow="What the data says"
      title="Nine calls in ten are a polite no"
      lede="Before any model, the shape of the problem: a heavy class imbalance, a fifth of the credit-default field missing outright, and a macroeconomic backdrop that moves conversion more than anything about the customer."
    >
      <div className="grid items-start gap-4 lg:grid-cols-[1fr_1.4fr]">
        <div className="flex flex-col gap-4">
          <div className="panel p-5">
            <h3 className="text-[0.9375rem]">Class balance</h3>
            <p className="mt-1 text-[0.8125rem] text-ink-3">
              {eda.positives.toLocaleString('en-US')} subscriptions out of{' '}
              {eda.rows.toLocaleString('en-US')} contacts.
            </p>
            <div className="mt-4 flex h-8 gap-[2px] overflow-hidden rounded-[4px]">
              <span
                className="grid place-items-center bg-white/10 text-[0.6875rem] text-ink-2"
                style={{ width: `${(1 - share) * 100}%` }}
              >
                {((1 - share) * 100).toFixed(1)}% no
              </span>
              <span
                className="bg-series-3"
                style={{ width: `${share * 100}%` }}
                title={`${(share * 100).toFixed(2)}% yes`}
              />
            </div>
            <p className="mt-3 text-[0.8125rem] leading-relaxed text-ink-2">
              A model that predicts &ldquo;no&rdquo; for everyone is{' '}
              <span className="text-ink">88.7% accurate</span> and worth nothing. Accuracy is not
              the metric here; ranking quality is.
            </p>
          </div>

          <div className="panel p-5">
            <h3 className="text-[0.9375rem]">Missing values</h3>
            <p className="mt-1 text-[0.8125rem] text-ink-3">
              Share of records coded <span className="font-mono">unknown</span>.
            </p>
            <ul className="mt-4 m-0 flex list-none flex-col gap-2.5 p-0">
              {eda.missingness.map((m) => (
                <li key={m.label} className="grid grid-cols-[5.5rem_1fr_2.75rem] items-center gap-3">
                  <span className="truncate text-right text-[0.8125rem] text-ink-2">{m.label}</span>
                  <span className="flex h-2.5 items-center">
                    <span
                      className="block h-full rounded-r-[3px] bg-series-4"
                      style={{ width: `${Math.max(m.pct * 4, m.pct > 0 ? 1.5 : 0)}%` }}
                    />
                  </span>
                  <span className="num text-right text-[0.75rem] text-ink-2 tabular-nums">
                    {m.pct}%
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-4 text-[0.8125rem] leading-relaxed text-ink-2">
              Credit default is unknown for a fifth of the book — plausibly not at random, since
              people rarely volunteer it. It is dropped rather than imputed. Housing and loan gaps
              are under 2.5% and those 990 rows are removed.
            </p>
          </div>
        </div>

        <ChartFrame
          title={`Conversion rate by ${active.label.toLowerCase()}`}
          subtitle="Share of contacts in each group that ended in a term-deposit subscription."
          note={active.finding}
        >
          <div className="mb-5 flex flex-wrap gap-1.5" role="group" aria-label="Choose a dimension">
            {DIMS.map((dim) => (
              <button
                key={dim.key}
                type="button"
                className="chip"
                aria-pressed={active.key === dim.key}
                onClick={() => setActive(dim)}
              >
                {dim.label}
              </button>
            ))}
          </div>
          {/* held at the tallest dimension's height so switching does not jump the page */}
          <div className="flex min-h-[19.5rem] flex-col justify-center">
            <RateBars rows={active.rows} baseline={eda.rate} raw={active.raw} />
          </div>
        </ChartFrame>
      </div>
    </Section>
  )
}
