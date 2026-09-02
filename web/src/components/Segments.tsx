import { Section } from './Section'
import { ChartFrame } from './charts/ChartFrame'
import { LineChart } from './charts/LineChart'
import segments from '../data/segments.json'
import { num, prettyLabel } from '../lib/ui'

const COPY: Record<string, { tag: string; body: string; action: string }> = {
  'Loyal Responders': {
    tag: 'Won before, and the rates are low',
    body: 'Reached on a mobile 22 days after a successful earlier campaign, while Euribor sat near 0.98%. Two calls on average. Nothing else in the book looks like this.',
    action: 'Call first, every time. This is the cheapest revenue available.',
  },
  'Promising Prospects': {
    tag: 'Cheap money, barely touched',
    body: 'The same low-rate window but no prior relationship — 0.3 previous contacts on average, two attempts this campaign. Nearly a third of the portfolio sits here.',
    action: 'The real growth pool. Worth a scripted second attempt, not a tenth.',
  },
  'Unengaged Majority': {
    tag: 'The high-rate default pool',
    body: 'Contacted while Euribor was 4.8% with no prior relationship. Almost two thirds of every record in the file, converting at well under half the baseline.',
    action: 'Suppress by default. This is where a blind dialler burns its budget.',
  },
  'Resistant Non-Responders': {
    tag: 'Over-dialled on a landline',
    body: 'Twelve and a half calls on average, on a telephone line, in July at the top of the rate cycle — and the shortest conversations in the dataset at 175 seconds.',
    action: 'Cap attempts. Every additional call here is a pure loss.',
  },
}

const TONE = ['var(--color-series-3)', 'var(--color-series-1)', 'var(--color-ink-3)', 'var(--color-ink-3)']

export function Segments() {
  const maxRate = Math.max(...segments.segments.map((s) => s.rate))

  return (
    <Section
      id="segments"
      eyebrow="Segmentation"
      title="Four personas, found without ever looking at the answer"
      lede={
        <>
          A ranking tells you who to ring. It does not tell you who these people are. K-Prototypes
          clusters the full {num(segments.total)} records on mixed numeric and categorical
          fields at once — no lossy one-hot, no k-means on dummies — and the target column is held
          back entirely, so conversion rate per cluster is an out-of-sample read on whether the
          segmentation found anything real.
        </>
      }
    >
      <div className="grid items-start gap-4 lg:grid-cols-[1fr_1.5fr]">
        <ChartFrame
          title="Choosing k"
          subtitle="Total dissimilarity against number of clusters, k = 2 to 8."
          note="The cost curve has no sharp elbow — it rarely does on data this mixed — so k = 4 is a judgement call: the last k that still yields segments a campaign manager can name and act on. Past it the clusters split on the same macro axis rather than describing new people."
        >
          <LineChart
            ariaLabel="Elbow plot of K-Prototypes cost against k, with k equals 4 marked"
            series={[
              {
                label: 'Cost',
                color: 'var(--color-series-1)',
                points: segments.elbow.map((e) => [e.k, e.cost] as [number, number]),
                markAt: 4,
              },
            ]}
            xDomain={[2, 8]}
            yDomain={[200000, 380000]}
            xTicks={[2, 3, 4, 5, 6, 7, 8]}
            yTicks={[200000, 250000, 300000, 350000]}
            xLabel="Number of clusters (k)"
            yLabel="Cost"
            format={(v, axis) => (axis === 'x' ? String(v) : `${Math.round(v / 1000)}k`)}
            height={250}
          />
        </ChartFrame>

        <ChartFrame
          title="Conversion by segment"
          subtitle={`Share of each cluster that subscribed, against the ${segments.baseline}% portfolio baseline.`}
          note="Clustering ran on all 41,188 records rather than subscribers only. That is deliberate: a 63% cluster only means something measured against the 11.3% it is not."
        >
          <ul className="m-0 flex list-none flex-col gap-3 p-0">
            {segments.segments.map((seg, i) => (
              <li key={seg.name} className="grid grid-cols-[1fr] gap-1.5">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-[0.8125rem] text-ink-2">{seg.name}</span>
                  <span className="num text-[0.8125rem] text-ink-3 tabular-nums">
                    {seg.lift}× baseline · {num(seg.size)} customers
                  </span>
                </div>
                <div className="relative flex h-5 items-center">
                  <span
                    className="block h-full rounded-r-[4px]"
                    style={{
                      width: `${(seg.rate / (maxRate * 1.15)) * 100}%`,
                      background: TONE[i],
                    }}
                  />
                  <span className="num ml-2 text-[0.8125rem] font-medium text-ink tabular-nums">
                    {seg.rate}%
                  </span>
                  <span
                    aria-hidden
                    className="pointer-events-none absolute inset-y-0 border-l border-dashed border-white/55"
                    style={{ left: `${(segments.baseline / (maxRate * 1.15)) * 100}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-4 flex items-center gap-2 text-[0.75rem] text-ink-3">
            <span aria-hidden className="inline-block h-3 border-l border-dashed border-white/55" />
            Portfolio baseline {segments.baseline}%
          </p>
        </ChartFrame>
      </div>

      <ol className="mt-4 m-0 grid list-none gap-px overflow-hidden rounded-xl border border-white/8 bg-white/8 p-0 md:grid-cols-2 xl:grid-cols-4">
        {segments.segments.map((seg, i) => {
          const copy = COPY[seg.name]
          return (
            <li key={seg.name} className="flex flex-col bg-panel p-5">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="text-[1.0625rem] leading-tight">{seg.name}</h3>
                <span
                  className="metric num shrink-0 text-[1.375rem] tabular-nums"
                  style={{ color: TONE[i] }}
                >
                  {seg.rate}%
                </span>
              </div>
              <p className="mt-1 min-h-[2.25rem] text-[0.75rem] leading-snug text-ink-3">
                {copy.tag}
              </p>
              <p className="mt-2 text-[0.8125rem] leading-relaxed text-ink-2">{copy.body}</p>

              <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-2 border-t border-white/8 pt-4 text-[0.75rem]">
                {[
                  ['Customers', num(seg.size)],
                  ['Share', `${seg.share}%`],
                  ['Mean age', seg.numeric.age.toFixed(0)],
                  ['Calls', seg.numeric.campaign.toFixed(1)],
                  ['Euribor', `${seg.numeric.euribor3m.toFixed(2)}%`],
                  ['Channel', prettyLabel(seg.modes.contact)],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <dt className="text-ink-3">{k}</dt>
                    <dd className="num m-0 text-right text-ink-2 tabular-nums">{v}</dd>
                  </div>
                ))}
              </dl>

              <p className="mt-4 border-t border-white/8 pt-3 text-[0.75rem] leading-relaxed text-ink-2">
                <span className="text-ink">Do:</span> {copy.action}
              </p>
            </li>
          )
        })}
      </ol>

      <div className="panel mt-4 p-5 sm:p-6">
        <h3 className="text-[1.0625rem]">What the clusters actually split on</h3>
        <p className="mt-3 max-w-4xl text-[0.875rem] leading-relaxed text-ink-2">
          Look down the four profiles and the demographics barely move: mean age 39–42 across every
          segment, the same modal job, the same modal education, the same marital status. What
          separates them is Euribor at 0.98% versus 4.90%, prior campaign contact, and how many
          times the number had already been dialled. K-Prototypes did not find four kinds of person
          — it rediscovered the interest-rate cycle and the campaign&rsquo;s own contact history,
          which is the same conclusion the classifier&rsquo;s split gains reached independently.
        </p>
        <p className="mt-3 max-w-4xl text-[0.875rem] leading-relaxed text-ink-2">
          That is a useful finding rather than a failed one, but it changes what the segmentation is
          for. These are not enduring customer personas to build a brand strategy around; they are{' '}
          <span className="text-ink">campaign states</span> — and the actionable half is contact
          policy: who to suppress, and when to stop dialling.
        </p>
      </div>
    </Section>
  )
}
