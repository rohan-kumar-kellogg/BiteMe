'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ArrowRight } from 'lucide-react'
import { BiteMeLogo, BiteMePieBackground } from '@/components/bite-me-logo'

interface WelcomeScreenProps {
  onSubmit: (identity: { username: string; email: string }) => void
  isLoading: boolean
  initialUsername?: string
  initialEmail?: string
}

export function WelcomeScreen({ onSubmit, isLoading, initialUsername = '', initialEmail = '' }: WelcomeScreenProps) {
  const [username, setUsername] = useState(initialUsername)
  const [email, setEmail] = useState(initialEmail)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (username.trim() && email.trim()) {
      onSubmit({ username: username.trim().toLowerCase(), email: email.trim().toLowerCase() })
    }
  }

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Large decorative pie background - blended, centered */}
      <div className="absolute top-[45%] left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
        <BiteMePieBackground size={605} className="md:hidden" />
        <BiteMePieBackground size={825} className="hidden md:block lg:hidden" />
        <BiteMePieBackground size={1045} className="hidden lg:block" />
      </div>
      
      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4">
        <BiteMeLogo iconSize={32} />
        <Button asChild variant="ghost" size="sm" className="rounded-full">
          <Link href="/model-card">Model Card</Link>
        </Button>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 flex-1 flex items-center justify-center px-4 py-8">
        <div className="w-full max-w-lg">
          <form
            onSubmit={handleSubmit}
            className="rounded-3xl border border-border/60 bg-card/90 backdrop-blur-sm shadow-xl shadow-black/5 p-6 md:p-8 space-y-5"
          >
            <div className="text-center space-y-2">
              <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">Welcome to BiteMe</h1>
              <p className="text-sm text-muted-foreground">Enter your identity to continue your flavor profile.</p>
            </div>
            <div className="space-y-3">
              <Input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="h-12 rounded-2xl bg-background"
                disabled={isLoading}
                autoFocus
              />
              <Input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="h-12 rounded-2xl bg-background"
                disabled={isLoading}
              />
            </div>
            <Button
              type="submit"
              size="lg"
              className="w-full h-12 rounded-2xl font-semibold group"
              disabled={!username.trim() || !email.trim() || isLoading}
            >
              {isLoading ? (
                <span className="flex items-center gap-3">
                  <span className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Continuing...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Continue
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </span>
              )}
            </Button>
            <p className="text-xs text-center text-muted-foreground">No password needed for this demo.</p>
          </form>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-8 text-center space-y-2">
        <p className="text-sm text-muted-foreground/80 italic">
          Judging your taste, one bite at a time.
        </p>
      </footer>
    </div>
  )
}
