import type { Row } from './lgbm'

/**
 * The three macroeconomic features move together — they are quarterly readings, not
 * customer attributes — so the UI offers the regimes that actually occurred rather
 * than three free sliders that would put the model off its training manifold.
 * Rates are the observed means for each employment level in the raw file.
 */
export const REGIMES = [
  {
    id: 'crisis',
    label: 'Crisis peak',
    period: '2008',
    historical: 5.3,
    note: 'Euribor near 5%, employment at its high. Deposits compete with everything.',
    values: { 'nr.employed': 5228.1, euribor3m: 4.95, 'cons.conf.idx': -40.4 },
  },
  {
    id: 'falling',
    label: 'Rates falling',
    period: '2009',
    historical: 12.8,
    note: 'Employment sliding, Euribor down to 1.3%. The campaign starts to work.',
    values: { 'nr.employed': 5099.1, euribor3m: 1.34, 'cons.conf.idx': -46.6 },
  },
  {
    id: 'cheap',
    label: 'Cheap money',
    period: '2010',
    historical: 42.4,
    note: 'Euribor under 1%. A term deposit is suddenly a reasonable place to sit.',
    values: { 'nr.employed': 5017.5, euribor3m: 0.74, 'cons.conf.idx': -28.7 },
  },
] as const

export const JOBS = [
  'admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
  'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown',
]
export const MARITAL = ['married', 'single', 'divorced', 'unknown']
export const EDUCATION = [
  'basic 4 years', 'basic 6 years', 'basic 9 years', 'high school',
  'professional course', 'university degree', 'illiterate', 'unknown',
]
export const MONTHS = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
export const DAYS = ['mon', 'tue', 'wed', 'thu', 'fri']

export const BASE_ROW: Row = {
  age: 38,
  campaign: 1,
  previous: 0,
  job: 'admin',
  marital: 'married',
  education: 'university degree',
  housing: 'yes',
  loan: 'no',
  contact: 'cellular',
  month: 'may',
  day_of_week: 'thu',
  poutcome: 'nonexistent',
  was_contacted_before: 'no',
  is_default_status_known: 'yes',
  ...REGIMES[1].values,
}

export const PRESETS: { id: string; label: string; blurb: string; row: Row }[] = [
  {
    id: 'typical',
    label: 'Typical cold call',
    blurb: 'The median record in the book: no history, first attempt, mid-campaign.',
    row: BASE_ROW,
  },
  {
    id: 'warm',
    label: 'Won before',
    blurb: 'Retiree who subscribed in an earlier campaign, called back in a low-rate month.',
    row: {
      ...BASE_ROW,
      age: 62,
      job: 'retired',
      marital: 'married',
      education: 'basic 4 years',
      housing: 'no',
      month: 'oct',
      previous: 2,
      poutcome: 'success',
      was_contacted_before: 'yes',
      ...REGIMES[2].values,
    },
  },
  {
    id: 'burned',
    label: 'Over-dialled',
    blurb: 'Eleventh attempt on a landline in May, at the top of the rate cycle.',
    row: {
      ...BASE_ROW,
      age: 44,
      job: 'blue-collar',
      education: 'basic 9 years',
      contact: 'telephone',
      month: 'may',
      campaign: 11,
      ...REGIMES[0].values,
    },
  },
]
