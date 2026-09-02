import { Section } from './Section'

const SHIP = [
  {
    t: 'Rank, then cut — do not classify',
    d: 'Ship the score, not the label. The dialler works down the ranked list until the day’s capacity runs out, which makes the operating point a capacity decision the campaign manager owns rather than a hyperparameter buried in a notebook.',
  },
  {
    t: 'Suppress the bottom half outright',
    d: 'The lowest five deciles of the test set contain 16.9% of subscribers between them. Removing them frees half the call budget for a sixth of the upside — the clearest win in the whole analysis.',
  },
  {
    t: 'Cap attempts at five',
    d: 'Conversion falls from 13.0% on the first call to 3.2% past the tenth, and the resistant cluster averages 12.4 attempts. An attempt cap needs no model at all and pays for itself immediately.',
  },
  {
    t: 'Retrain on the rate cycle, not the calendar',
    d: 'Half the model’s split gain sits in quarterly macro indicators. Tie retraining to moves in Euribor and employment rather than to a fixed monthly job, and alert on drift in those two inputs specifically.',
  },
]

const WATCH = [
  {
    t: 'The split is random, not temporal',
    d: 'Train and test are drawn from the same quarters, so the macro features are effectively shared between them. A forward-chained split — train on 2008–09, test on 2010 — would give a lower and more honest number. I would run that before promising anything to a business.',
  },
  {
    t: 'The model has learned a regime as much as a person',
    d: 'nr.employed alone carries 50.6% of split gain. Deployed into a rate environment unlike anything in 2008–2010, the ranking would degrade in ways this test set cannot show.',
  },
  {
    t: 'The economics are assumed',
    d: 'The dataset carries no cost or margin figures. Every euro in the threshold lab comes from inputs you can edit; the shape of the trade-off is real, the absolute numbers are illustrative.',
  },
  {
    t: 'One dataset, one bank, one country',
    d: 'Portuguese retail deposits between 2008 and 2010. Nothing here should be assumed to transfer to another market without revalidation.',
  },
]

function List({ items, accent }: { items: { t: string; d: string }[]; accent: string }) {
  return (
    <ol className="m-0 flex list-none flex-col gap-5 p-0">
      {items.map((item, i) => (
        <li key={item.t} className="grid grid-cols-[1.5rem_1fr] gap-3">
          <span
            aria-hidden
            className="num mt-0.5 text-[0.75rem] tabular-nums"
            style={{ color: accent }}
          >
            {String(i + 1).padStart(2, '0')}
          </span>
          <div>
            <h3 className="text-[0.9375rem] leading-snug">{item.t}</h3>
            <p className="mt-1.5 text-[0.875rem] leading-relaxed text-ink-2">{item.d}</p>
          </div>
        </li>
      ))}
    </ol>
  )
}

export function Closing() {
  return (
    <Section
      id="closing"
      eyebrow="Recommendations"
      title="What I would ship, and what I would keep watching"
      lede="A model is a proposal about how to spend a budget. These are the four changes the analysis supports, and the four reasons I would not oversell it."
    >
      <div className="grid gap-px overflow-hidden rounded-xl border border-white/8 bg-white/8 lg:grid-cols-2">
        <div className="bg-panel p-6 sm:p-8">
          <h3 className="eyebrow mb-6">Ship</h3>
          <List items={SHIP} accent="var(--color-series-3)" />
        </div>
        <div className="bg-panel p-6 sm:p-8">
          <h3 className="eyebrow mb-6">Watch</h3>
          <List items={WATCH} accent="var(--color-warning)" />
        </div>
      </div>
    </Section>
  )
}
