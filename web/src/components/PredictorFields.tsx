import type { ReactNode } from 'react'
import { prettyLabel } from '../lib/ui'

export function Field({
  label,
  htmlFor,
  children,
}: {
  label: string
  htmlFor: string
  children: ReactNode
}) {
  return (
    <div>
      <label htmlFor={htmlFor} className="block text-[0.75rem] text-ink-3">
        {label}
      </label>
      <div className="mt-1.5">{children}</div>
    </div>
  )
}

export function Select({
  label,
  id,
  value,
  options,
  onChange,
  disabled = false,
}: {
  label: string
  id: string
  value: string
  options: string[]
  onChange: (value: string) => void
  disabled?: boolean
}) {
  return (
    <Field label={label} htmlFor={id}>
      <div className="relative">
        <select
          id={id}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className="field disabled:cursor-not-allowed disabled:text-ink-3"
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {prettyLabel(option)}
            </option>
          ))}
        </select>
        <span
          aria-hidden
          className="pointer-events-none absolute top-1/2 right-2.5 -translate-y-1/2 text-[0.625rem] text-ink-3"
        >
          ▼
        </span>
      </div>
    </Field>
  )
}
