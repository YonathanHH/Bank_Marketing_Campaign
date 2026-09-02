import { useEffect, useRef, useState } from 'react'

export const num = (v: number) => v.toLocaleString('en-US')

export const eur = (v: number) =>
  v >= 1_000_000
    ? `€${(v / 1_000_000).toFixed(2)}M`
    : v >= 1_000
      ? `€${Math.round(v / 1_000)}K`
      : `€${Math.round(v)}`

const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

const MONTHS: Record<string, string> = {
  jan: 'Jan', feb: 'Feb', mar: 'Mar', apr: 'Apr', may: 'May', jun: 'Jun',
  jul: 'Jul', aug: 'Aug', sep: 'Sep', oct: 'Oct', nov: 'Nov', dec: 'Dec',
}
const DAYS: Record<string, string> = {
  mon: 'Monday', tue: 'Tuesday', wed: 'Wednesday', thu: 'Thursday', fri: 'Friday',
}

/** Turn dataset codes into something a person would write. Numbers pass through. */
export function prettyLabel(raw: string) {
  if (MONTHS[raw]) return MONTHS[raw]
  if (DAYS[raw]) return DAYS[raw]
  if (raw !== '' && !Number.isNaN(Number(raw))) return raw
  return titleCase(raw.replace(/[._]/g, ' '))
}

/**
 * Reveal-on-scroll. Starts already-shown where IntersectionObserver is missing, so
 * content is never trapped behind an observer that will not fire.
 */
export function useReveal<T extends HTMLElement>(rootMargin = '-8% 0px -8% 0px') {
  const ref = useRef<T>(null)
  const [shown, setShown] = useState(() => typeof IntersectionObserver === 'undefined')

  useEffect(() => {
    const node = ref.current
    if (!node || typeof IntersectionObserver === 'undefined') return

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true)
          io.disconnect()
        }
      },
      { rootMargin },
    )
    io.observe(node)
    return () => io.disconnect()
  }, [rootMargin])

  return { ref, shown }
}

/** Measured width of an element, so a chart can draw one SVG unit per CSS pixel. */
export function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(fallback)

  useEffect(() => {
    const node = ref.current
    if (!node || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    ro.observe(node)
    return () => ro.disconnect()
  }, [])

  return { ref, width }
}
