import { Section } from './Section'
import { ChartFrame } from './charts/ChartFrame'
import { LineChart } from './charts/LineChart'
import metrics from '../data/metrics.json'
import eda from '../data/eda.json'
import { prettyLabel } from '../lib/ui'

const STEPS = [
  { n: 'Clean', d: 'unknown → NA · drop the 20.8%-missing default column · drop 990 rows missing housing or loan' },
  { n: 'Engineer', d: 'pdays 999 → was_contacted_before · tidy job and education labels' },
  { n: 'Encode', d: 'StandardScaler on 6 numerics · one-hot with drop-first on 11 categoricals → 46 columns' },
  { n: 'Fit', d: 'LightGBM, class_weight=balanced, 250 trees, depth 7, lr 0.03' },
  { n: 'Select', d: 'GridSearchCV on 5-fold stratified F2 — recall weighted 4× precision' },
]

const IMPORTANCE_NAMES: Record<string, string> = {
  'nr.employed': 'Employment level (quarterly)',
  euribor3m: 'Euribor 3-month rate',
  'cons.conf.idx': 'Consumer confidence index',
  age: 'Age',
  campaign: 'Calls made this campaign',
  previous: 'Prior campaign contacts',
}

function readable(raw: string) {
  const [, kind, rest] = raw.match(/^(num|cat)__(.+)$/) ?? []
  if (kind === 'num') return IMPORTANCE_NAMES[rest] ?? rest
  const cut = rest.lastIndexOf('_')
  const field = rest.slice(0, cut)
  const level = rest.slice(cut + 1)
  const FIELDS: Record<string, string> = {
    month: 'Month',
    contact: 'Channel',
    was_contacted_before: 'Contacted before',
    poutcome: 'Previous outcome',
    job: 'Occupation',
    education: 'Education',
    day_of_week: 'Day',
    marital: 'Marital status',
    housing: 'Housing loan',
    loan: 'Personal loan',
  }
  return `${FIELDS[field] ?? field}: ${prettyLabel(level)}`
}

export function ModelSection() {
  const ranked = [...metrics.benchmark].sort((a, b) => b.rocAuc - a.rocAuc)
  const maxGain = metrics.importance[0].gain

  return (
    <Section
      id="model"
      eyebrow="The model"
      title="Gradient boosting, tuned for the mistake that actually costs money"
      lede={
        <>
          Five candidates on identical folds, scored on F2 rather than accuracy. The winner is a
          class-balanced LightGBM trained on{' '}
          <span className="num text-ink tabular-nums">
            {metrics.train.rows.toLocaleString('en-US')}
          </span>{' '}
          contacts and measured once, at the end, on{' '}
          <span className="num text-ink tabular-nums">
            {metrics.test.rows.toLocaleString('en-US')}
          </span>{' '}
          it had never seen.
        </>
      }
    >
      <ol className="m-0 grid list-none gap-px overflow-hidden rounded-xl border border-white/8 bg-white/8 p-0 sm:grid-cols-2 lg:grid-cols-5">
        {STEPS.map((step, i) => (
          <li key={step.n} className="bg-panel p-4">
            <div className="flex items-baseline gap-2">
              <span className="eyebrow">{String(i + 1).padStart(2, '0')}</span>
              <h3 className="text-[0.9375rem]">{step.n}</h3>
            </div>
            <p className="mt-2 text-[0.75rem] leading-relaxed text-ink-3">{step.d}</p>
          </li>
        ))}
      </ol>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1.15fr_1fr]">
        <figure className="panel m-0 p-4 sm:p-5">
          <figcaption className="mb-4">
            <h3 className="text-[0.9375rem]">Candidate comparison</h3>
            <p className="mt-1 text-[0.8125rem] text-ink-3">
              Held-out test set, ranked by ROC-AUC. F2 is measured at a 0.50 cut-off.
            </p>
          </figcaption>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[30rem] border-separate border-spacing-0 text-[0.875rem]">
              <thead>
                <tr>
                  {['Model', 'ROC-AUC', 'PR-AUC', 'Recall', 'F2'].map((h, i) => (
                    <th
                      key={h}
                      scope="col"
                      className={`p-2 pb-2.5 text-[0.75rem] font-medium text-ink-3 ${i === 0 ? 'text-left' : 'text-right'}`}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ranked.map((row) => {
                  const won = 'deployed' in row && row.deployed
                  return (
                    <tr key={row.model} className={won ? 'bg-series-1/10' : undefined}>
                      <th
                        scope="row"
                        className={`border-t border-white/8 p-2 text-left font-normal ${won ? 'text-ink' : 'text-ink-2'}`}
                      >
                        {won && (
                          <span aria-hidden className="mr-1.5 text-series-1">
                            ▸
                          </span>
                        )}
                        {row.model}
                      </th>
                      {[row.rocAuc, row.prAuc, row.recall, row.f2].map((v, i) => (
                        <td
                          key={i}
                          className={`num border-t border-white/8 p-2 text-right tabular-nums ${won ? 'text-ink' : 'text-ink-3'}`}
                        >
                          {v.toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-[0.75rem] leading-relaxed text-ink-3">
            Gradient boosting edges LightGBM on AUC by 0.002 — inside the noise of a single split —
            but with no class weighting it puts almost no mass above 0.50, finding 24.6% of
            subscribers where LightGBM finds 64.9%. Random forest lands in the same place despite
            balanced weights. Ranking quality and a usable operating point are different things, and
            a campaign needs both.
          </p>
        </figure>

        <ChartFrame
          title="Precision–recall, held-out test set"
          subtitle={`Precision against recall for every threshold. Chance sits at the ${eda.rate}% base rate.`}
          legend={[
            { label: 'Deployed model', color: 'var(--color-series-1)' },
            { label: 'No-skill baseline', color: 'var(--color-ink-3)', dashed: true },
          ]}
          note={`PR-AUC of ${metrics.headline.prAuc.toFixed(3)} against a ${(eda.rate / 100).toFixed(3)} no-skill floor. On a class this rare, the PR curve is the honest picture — ROC flatters any model when negatives dominate.`}
        >
          <LineChart
            ariaLabel={`Precision-recall curve, area under curve ${metrics.headline.prAuc}`}
            series={[
              {
                label: 'Deployed model',
                color: 'var(--color-series-1)',
                points: metrics.pr as [number, number][],
              },
              {
                label: 'No-skill baseline',
                color: 'var(--color-ink-3)',
                points: [
                  [0, eda.rate / 100],
                  [1, eda.rate / 100],
                ],
                dashed: true,
              },
            ]}
            xDomain={[0, 1]}
            yDomain={[0, 1]}
            xTicks={[0, 0.25, 0.5, 0.75, 1]}
            yTicks={[0, 0.25, 0.5, 0.75, 1]}
            xLabel="Recall — share of subscribers found"
            yLabel="Precision"
            format={(v) => v.toFixed(2)}
            height={300}
          />
        </ChartFrame>
      </div>

      <ChartFrame
        className="mt-4"
        title="Where the splits happen"
        subtitle="Share of total split gain per encoded feature, top 14 of 46."
        note={
          <>
            <span className="text-ink-2">Read this as a warning, not a trophy.</span> Half the
            model&rsquo;s gain sits in <span className="font-mono">nr.employed</span> — the
            quarterly employment level — with Euribor and consumer confidence close behind. These are
            macroeconomic conditions, identical for every customer contacted in the same quarter.
            The model has partly learned <em>when</em> a good campaign ran rather than{' '}
            <em>who</em> answers yes, which is why it needs retraining as the rate cycle turns.
          </>
        }
      >
        <ul className="m-0 flex list-none flex-col gap-2 p-0">
          {metrics.importance.map((item) => (
            <li
              key={item.feature}
              className="grid grid-cols-[9.5rem_1fr_3rem] items-center gap-3 sm:grid-cols-[14rem_1fr_3rem]"
            >
              <span className="truncate text-right text-[0.8125rem] text-ink-2">
                {readable(item.feature)}
              </span>
              <span className="flex h-4 items-center">
                <span
                  className="block h-full rounded-r-[4px] bg-series-1"
                  style={{ width: `${(item.gain / maxGain) * 100}%` }}
                />
              </span>
              <span className="num text-right text-[0.8125rem] text-ink tabular-nums">
                {item.gain.toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      </ChartFrame>
    </Section>
  )
}
