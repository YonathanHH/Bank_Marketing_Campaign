import { Section } from './Section'
import { ChartFrame } from './charts/ChartFrame'
import { LineChart } from './charts/LineChart'
import eda from '../data/eda.json'
import metrics from '../data/metrics.json'

const { withDuration, withoutDuration } = metrics.leakage

const ROWS = [
  { key: 'rocAuc' as const, label: 'ROC-AUC', fmt: (v: number) => v.toFixed(3) },
  { key: 'recall' as const, label: 'Recall on subscribers', fmt: (v: number) => `${(v * 100).toFixed(1)}%` },
  { key: 'f2' as const, label: 'F2-score', fmt: (v: number) => v.toFixed(3) },
  { key: 'prAuc' as const, label: 'PR-AUC', fmt: (v: number) => v.toFixed(3) },
]

export function LeakageSection() {
  return (
    <Section
      id="leakage"
      eyebrow="The result I threw away"
      title="A 0.95 AUC that would have failed on day one"
      lede={
        <>
          The dataset ships a column called <span className="font-mono text-ink">duration</span> —
          how many seconds the call lasted. Subscribers talked for{' '}
          <span className="num text-ink tabular-nums">{eda.durationByClass.yes}s</span> on average;
          everyone else hung up after{' '}
          <span className="num text-ink tabular-nums">{eda.durationByClass.no}s</span>. It is by far
          the strongest predictor in the file, and it is unusable.
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <div className="flex flex-col gap-4">
          <div className="panel p-5 sm:p-6">
            <h3 className="text-[1.0625rem]">Why it cannot be used</h3>
            <p className="mt-3 text-[0.875rem] leading-relaxed text-ink-2">
              A call&rsquo;s duration is only known once the call has ended. A model that needs it
              in order to decide whether to place the call is asking for the answer before it will
              give you one. The UCI documentation says so outright — the column exists to benchmark
              algorithms, not to build call lists.
            </p>
            <p className="mt-3 text-[0.875rem] leading-relaxed text-ink-2">
              It is also a proxy for the outcome rather than a cause of it. Long calls do not create
              subscribers; interested people stay on the line. Train on it and you learn to predict
              interest from interest.
            </p>
            <div className="mt-5 rounded-lg border border-white/8 bg-white/[0.03] p-4">
              <p className="m-0 text-[0.8125rem] leading-relaxed text-ink-2">
                <span className="text-ink">The test.</span> Same LightGBM configuration, same split,
                same seed. The only difference is one column.
              </p>
            </div>
          </div>

          <table className="panel w-full border-separate border-spacing-0 text-[0.875rem]">
            <caption className="sr-only">
              Held-out test metrics with and without the call-duration feature
            </caption>
            <thead>
              <tr>
                <th scope="col" className="p-3 pb-2 text-left text-[0.75rem] font-medium text-ink-3">
                  Held-out metric
                </th>
                <th scope="col" className="p-3 pb-2 text-right text-[0.75rem] font-medium text-series-2">
                  With duration
                </th>
                <th scope="col" className="p-3 pb-2 text-right text-[0.75rem] font-medium text-series-1">
                  Deployed
                </th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.key}>
                  <th
                    scope="row"
                    className="border-t border-white/8 p-3 text-left font-normal text-ink-2"
                  >
                    {row.label}
                  </th>
                  <td className="num border-t border-white/8 p-3 text-right text-ink-3 tabular-nums">
                    {row.fmt(withDuration[row.key])}
                  </td>
                  <td className="num border-t border-white/8 p-3 text-right font-medium text-ink tabular-nums">
                    {row.fmt(withoutDuration[row.key])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="m-0 text-[0.875rem] leading-relaxed text-ink-2">
            The leaky model looks dramatically better on every line. It is not better — it is
            answering a different, useless question. Every number elsewhere on this page comes from
            the {withoutDuration.rocAuc.toFixed(3)} model.
          </p>
        </div>

        <ChartFrame
          title="ROC curves, held-out test set"
          subtitle="The gap between the two curves is the entire value of the feature you are not allowed to have."
          legend={[
            { label: 'With duration (rejected)', color: 'var(--color-series-2)' },
            { label: 'Deployed model', color: 'var(--color-series-1)' },
            { label: 'Random', color: 'var(--color-ink-3)', dashed: true },
          ]}
          note={
            <>
              Area under the curve falls from {withDuration.rocAuc.toFixed(3)} to{' '}
              {withoutDuration.rocAuc.toFixed(3)} once the leak is closed. Reporting the first number
              would have been the easiest way to look good in a portfolio and the fastest way to
              lose trust in production.
            </>
          }
          className="lg:self-start"
        >
          <LineChart
            ariaLabel={`ROC curves. With call duration the area under the curve is ${withDuration.rocAuc}; without it, ${withoutDuration.rocAuc}.`}
            series={[
              {
                label: 'Deployed model',
                color: 'var(--color-series-1)',
                points: metrics.roc as [number, number][],
              },
              {
                label: 'With duration',
                color: 'var(--color-series-2)',
                points: metrics.leakage.roc as [number, number][],
              },
              {
                label: 'Random',
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
            xLabel="False positive rate — wasted calls"
            yLabel="True positive rate"
            format={(v) => v.toFixed(2)}
            height={330}
          />
        </ChartFrame>
      </div>
    </Section>
  )
}
