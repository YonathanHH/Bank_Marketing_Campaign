const REPO = 'https://github.com/YonathanHH/Bank_Marketing_Campaign'

const ARTEFACTS = [
  { label: 'Classification notebook', href: `${REPO}/blob/main/Final_Project_Alpha.ipynb` },
  { label: 'K-Prototypes notebook', href: `${REPO}/blob/main/clustering_kprototypes.ipynb` },
  { label: 'AutoML comparison', href: `${REPO}/blob/main/end_to_end_automl.ipynb` },
  { label: 'Model export script', href: `${REPO}/blob/main/tools/export_model.py` },
  { label: 'Streamlit app', href: `${REPO}/blob/main/app.py` },
]

const STACK = [
  ['Modelling', 'Python · scikit-learn · LightGBM · kmodes · SHAP'],
  ['This page', 'React · TypeScript · Vite · Tailwind · hand-built SVG charts'],
  ['Inference', '250-tree booster exported to JSON, traversed client-side'],
]

export function Footer() {
  return (
    <footer className="border-t border-white/8 py-16">
      <div className="shell">
        <div className="grid gap-10 lg:grid-cols-[1.2fr_1fr_1fr]">
          <div>
            <h2 className="text-[1.25rem]">Yonathan Hary Hutagalung</h2>
            <p className="mt-3 max-w-sm text-[0.875rem] leading-relaxed text-ink-2">
              Data scientist working on applied prediction problems — the kind where the model has
              to survive contact with a budget, a deadline and someone who will ask why the number
              moved.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <a
                href={REPO}
                target="_blank"
                rel="noreferrer"
                className="rounded-lg bg-ink px-4 py-2 text-[0.875rem] font-medium text-plane no-underline transition-opacity hover:opacity-88"
              >
                View the repository
              </a>
              <a
                href="https://github.com/YonathanHH"
                target="_blank"
                rel="noreferrer"
                className="rounded-lg border border-white/15 px-4 py-2 text-[0.875rem] text-ink-2 no-underline transition-colors hover:border-white/30 hover:text-ink"
              >
                GitHub profile
              </a>
            </div>
          </div>

          <nav aria-labelledby="artefacts">
            <h3 id="artefacts" className="eyebrow">
              Source artefacts
            </h3>
            <ul className="mt-4 m-0 flex list-none flex-col gap-2.5 p-0">
              {ARTEFACTS.map((item) => (
                <li key={item.label}>
                  <a
                    href={item.href}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[0.875rem] text-ink-2 no-underline transition-colors hover:text-ink"
                  >
                    {item.label} <span aria-hidden>↗</span>
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          <div>
            <h3 className="eyebrow">Built with</h3>
            <dl className="mt-4 m-0 flex flex-col gap-3">
              {STACK.map(([k, v]) => (
                <div key={k}>
                  <dt className="text-[0.75rem] text-ink-3">{k}</dt>
                  <dd className="m-0 mt-0.5 text-[0.8125rem] text-ink-2">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <div className="mt-12 flex flex-wrap items-center justify-between gap-4 border-t border-white/8 pt-6 text-[0.75rem] text-ink-3">
          <p className="m-0 max-w-2xl leading-relaxed">
            Data: Moro, S., Cortez, P. &amp; Rita, P. (2014).{' '}
            <em>A Data-Driven Approach to Predict the Success of Bank Telemarketing.</em> Decision
            Support Systems 62, 22–31.{' '}
            <a
              href="https://archive.ics.uci.edu/dataset/222/bank+marketing"
              target="_blank"
              rel="noreferrer"
              className="text-ink-2 no-underline hover:text-ink"
            >
              UCI Machine Learning Repository ↗
            </a>
          </p>
          <p className="m-0">MIT licensed</p>
        </div>
      </div>
    </footer>
  )
}
