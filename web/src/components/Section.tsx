import type { ReactNode } from 'react'
import { useReveal } from '../lib/ui'

type Props = {
  id: string
  eyebrow: string
  title: string
  lede?: ReactNode
  children: ReactNode
  /** Draw the hairline that separates this section from the one above. */
  divider?: boolean
}

export function Section({ id, eyebrow, title, lede, children, divider = true }: Props) {
  const { ref, shown } = useReveal<HTMLElement>()

  return (
    <section
      id={id}
      ref={ref}
      data-shown={shown}
      className={`reveal scroll-mt-20 py-16 sm:py-24 ${divider ? 'border-t border-white/8' : ''}`}
    >
      <div className="shell">
        <header className="max-w-2xl">
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="mt-3 text-[1.75rem] leading-[1.15] sm:text-[2.25rem]">{title}</h2>
          {lede && <div className="mt-4 text-[0.9375rem] leading-relaxed text-ink-2">{lede}</div>}
        </header>
        <div className="mt-10">{children}</div>
      </div>
    </section>
  )
}
