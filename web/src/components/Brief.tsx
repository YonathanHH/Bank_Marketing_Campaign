import { Section } from './Section'
import eda from '../data/eda.json'

const CARDS = [
  {
    n: '01',
    title: 'The cost is the call, not the customer',
    body: 'Every number on the list costs an agent minute whether or not it converts. At an 11% base rate, roughly nine in ten calls are pure expense. The lever is not persuasion — it is call order.',
  },
  {
    n: '02',
    title: 'A missed subscriber costs more than a wasted call',
    body: 'Skipping someone who would have said yes forfeits a whole deposit relationship; ringing someone who says no costs a few minutes. That asymmetry is why the model is tuned on F2, which weights recall four times as heavily as precision.',
  },
  {
    n: '03',
    title: 'The model has to work before the call',
    body: 'Anything the agent learns during the conversation is off-limits as a feature. That single constraint removes the dataset’s strongest predictor and cuts the headline score — and it is the difference between a demo and something a call centre can run.',
  },
]

export function Brief() {
  return (
    <Section
      id="brief"
      eyebrow="The brief"
      title="A call centre with 41,188 numbers and no idea which ones to ring"
      lede={
        <>
          The UCI Bank Marketing dataset records a Portuguese bank&rsquo;s direct telephone
          campaigns for term deposits between May 2008 and November 2010 — through the financial
          crisis and the collapse in interbank rates that followed. Of{' '}
          <span className="num text-ink tabular-nums">{eda.rows.toLocaleString('en-US')}</span>{' '}
          usable contacts, <span className="num text-ink tabular-nums">{eda.rate}%</span> ended in a
          subscription.
        </>
      }
      divider={false}
    >
      <ol className="m-0 grid list-none gap-px overflow-hidden rounded-xl border border-white/8 bg-white/8 p-0 md:grid-cols-3">
        {CARDS.map((card) => (
          <li key={card.n} className="bg-panel p-6">
            <span className="eyebrow">{card.n}</span>
            <h3 className="mt-3 text-[1.0625rem] leading-snug">{card.title}</h3>
            <p className="mt-3 text-[0.875rem] leading-relaxed text-ink-2">{card.body}</p>
          </li>
        ))}
      </ol>

      <div className="mt-6 grid gap-4 sm:grid-cols-[1fr_auto] sm:items-center">
        <p className="m-0 max-w-2xl text-[0.875rem] leading-relaxed text-ink-3">
          Two questions follow from that brief, and this page answers both: <span className="text-ink-2">who
          should we ring next</span> — a supervised ranking problem — and{' '}
          <span className="text-ink-2">who are these people</span> — an unsupervised segmentation
          problem on mixed numeric and categorical data.
        </p>
        <div className="flex gap-2 text-[0.6875rem]">
          <span className="chip">LightGBM</span>
          <span className="chip">K-Prototypes</span>
          <span className="chip">SHAP</span>
        </div>
      </div>
    </Section>
  )
}
