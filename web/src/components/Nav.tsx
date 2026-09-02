import { useEffect, useState } from 'react'

const LINKS = [
  { id: 'brief', label: 'Brief' },
  { id: 'data', label: 'Data' },
  { id: 'leakage', label: 'Leakage' },
  { id: 'model', label: 'Model' },
  { id: 'threshold', label: 'Threshold lab' },
  { id: 'predictor', label: 'Try it' },
  { id: 'segments', label: 'Segments' },
]

export function Nav() {
  const [active, setActive] = useState('')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const sections = LINKS.map((l) => document.getElementById(l.id)).filter(
      (el): el is HTMLElement => el !== null,
    )

    const onScroll = () => {
      const scrollable = document.body.scrollHeight - window.innerHeight
      setProgress(scrollable > 0 ? Math.min(1, window.scrollY / scrollable) : 0)

      // the section whose top has most recently passed the header
      const line = window.scrollY + 120
      let current = ''
      for (const el of sections) {
        if (el.offsetTop <= line) current = el.id
      }
      setActive(current)
    }

    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <div className="border-b border-white/8 bg-plane/85 backdrop-blur-md">
        <div className="shell flex h-14 items-center justify-between gap-6">
          <a href="#top" className="group flex items-center gap-2.5 text-ink no-underline">
            <span
              aria-hidden
              className="grid h-6 w-6 place-items-center rounded-[5px] bg-series-1 text-[0.6875rem] font-bold text-white"
            >
              Y
            </span>
            <span className="text-[0.8125rem] font-medium tracking-tight">
              <span className="sm:hidden">Telemarketing model</span>
              <span className="hidden sm:inline">Bank Telemarketing Propensity</span>
            </span>
          </a>

          <nav aria-label="Sections" className="hidden items-center gap-1 lg:flex">
            {LINKS.map((link) => (
              <a
                key={link.id}
                href={`#${link.id}`}
                aria-current={active === link.id ? 'true' : undefined}
                className={`rounded-md px-2.5 py-1.5 text-[0.8125rem] no-underline transition-colors ${
                  active === link.id ? 'text-ink' : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                {link.label}
              </a>
            ))}
          </nav>

          <a
            href="https://github.com/YonathanHH/Bank_Marketing_Campaign"
            target="_blank"
            rel="noreferrer"
            className="shrink-0 rounded-md border border-white/12 px-3 py-1.5 text-[0.8125rem] text-ink-2 no-underline transition-colors hover:border-white/25 hover:text-ink"
          >
            Source
          </a>
        </div>
      </div>
      <div
        aria-hidden
        className="h-px origin-left bg-series-1"
        style={{ transform: `scaleX(${progress})` }}
      />
    </header>
  )
}
