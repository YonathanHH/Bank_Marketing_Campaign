import type { ReactNode } from 'react'

export type LegendItem = { label: string; color: string; dashed?: boolean }

type Props = {
  title: string
  subtitle?: string
  legend?: LegendItem[]
  note?: ReactNode
  children: ReactNode
  className?: string
}

/**
 * Shared chart chrome: heading, subtitle, legend and footnote. A legend is always
 * rendered for two or more series so identity never rests on colour alone.
 */
export function ChartFrame({ title, subtitle, legend, note, children, className = '' }: Props) {
  return (
    <figure className={`panel m-0 p-4 sm:p-5 ${className}`}>
      <figcaption className="mb-4 flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
        <div className="min-w-0">
          <h3 className="text-[0.9375rem] leading-tight font-semibold">{title}</h3>
          {subtitle && <p className="mt-1 max-w-prose text-[0.8125rem] text-ink-3">{subtitle}</p>}
        </div>
        {legend && legend.length > 1 && (
          <ul className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1.5">
            {legend.map((item) => (
              <li key={item.label} className="flex items-center gap-2 text-[0.75rem] text-ink-2">
                <span
                  aria-hidden
                  className="inline-block h-0.5 w-4 rounded-full"
                  style={
                    item.dashed
                      ? {
                          backgroundImage: `repeating-linear-gradient(90deg, ${item.color} 0 4px, transparent 4px 7px)`,
                        }
                      : { background: item.color }
                  }
                />
                {item.label}
              </li>
            ))}
          </ul>
        )}
      </figcaption>
      {children}
      {note && <p className="mt-4 text-[0.75rem] leading-relaxed text-ink-3">{note}</p>}
    </figure>
  )
}
