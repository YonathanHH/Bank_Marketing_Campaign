import { useEffect } from 'react'
import { Brief } from './components/Brief'
import { Closing } from './components/Closing'
import { DataSection } from './components/DataSection'
import { Footer } from './components/Footer'
import { Hero } from './components/Hero'
import { LeakageSection } from './components/LeakageSection'
import { ModelSection } from './components/ModelSection'
import { Nav } from './components/Nav'
import { Predictor } from './components/Predictor'
import { Segments } from './components/Segments'
import { ThresholdLab } from './components/ThresholdLab'

export default function App() {
  /**
   * A deep link scrolls before React has laid the sections out, so the browser
   * lands in the wrong place (or past the end of a one-screen document). Re-run
   * the jump once the first frame is painted.
   */
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1))
    if (!id) return
    const frame = requestAnimationFrame(() =>
      document.getElementById(id)?.scrollIntoView({ behavior: 'auto', block: 'start' }),
    )
    return () => cancelAnimationFrame(frame)
  }, [])

  return (
    <>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-100 focus:rounded-md focus:bg-ink focus:px-3 focus:py-2 focus:text-[0.875rem] focus:text-plane"
      >
        Skip to content
      </a>
      <Nav />
      <main id="main">
        <Hero />
        <Brief />
        <DataSection />
        <LeakageSection />
        <ModelSection />
        <ThresholdLab />
        <Predictor />
        <Segments />
        <Closing />
      </main>
      <Footer />
    </>
  )
}
