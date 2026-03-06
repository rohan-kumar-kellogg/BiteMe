'use client'

import { useRef, useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, LogOut, Share2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { BiteMeLogo } from '@/components/bite-me-logo'
import { ProfileHero } from '@/components/profile/profile-hero'
import { FlavorFingerprint } from '@/components/profile/flavor-fingerprint'
import { TasteAnalysis } from '@/components/profile/taste-analysis'
import { CompatibleUsers } from '@/components/profile/compatible-users'
import { RecentUploads } from '@/components/profile/recent-uploads'
import { ClickedRecommendations } from '@/components/profile/clicked-recommendations'
import { UploadModule } from '@/components/profile/upload-module'
import { TasteTrends } from '@/components/profile/taste-trends'
import type { UserProfile } from '@/lib/types'

interface ProfilePageProps {
  profile: UserProfile
  fingerprintRefreshToken?: number
  onBack: () => void
  onLogout: () => void
  onUpload: (file: File) => Promise<void>
  onInviteToEat: (payload: {
    to_username: string
    to_email?: string
    restaurant_name?: string
    date?: string
    time?: string
    message?: string
  }) => Promise<void>
  onRecommendationClick: (dishLabel: string, cuisine?: string) => Promise<void>
  onRemoveRecommendationClick: (eventId: string) => Promise<void>
  onRemoveUpload: (uploadId: string) => Promise<void>
  isUploading: boolean
}

export function ProfilePage({
  profile,
  fingerprintRefreshToken = 0,
  onBack,
  onLogout,
  onUpload,
  onInviteToEat,
  onRecommendationClick,
  onRemoveRecommendationClick,
  onRemoveUpload,
  isUploading,
}: ProfilePageProps) {
  const [showTrends, setShowTrends] = useState(false)
  const [activeView, setActiveView] = useState<'profile' | 'matches'>('profile')
  const uploadRef = useRef<HTMLDivElement | null>(null)
  const hasUploads = profile.total_uploads > 0 || profile.recent_uploads.length > 0

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `${profile.username}'s Flavor Fingerprint`,
          text: `I'm ${profile.archetype}! Check out my taste profile.`,
          url: window.location.href,
        })
      } catch (err) {
        // User cancelled or error
      }
    }
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border/50">
        <div className="max-w-3xl mx-auto px-4 h-14 flex items-center justify-between">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onBack}
            className="gap-2 -ml-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="hidden sm:inline">Back</span>
          </Button>
          
          <BiteMeLogo iconSize={28} />
          
          <div className="flex items-center gap-1">
            <div className="flex items-center rounded-full border border-border/50 p-0.5 mr-1">
              <Button
                variant={activeView === 'profile' ? 'default' : 'ghost'}
                size="sm"
                className="h-8 rounded-full px-2 sm:px-3"
                onClick={() => setActiveView('profile')}
              >
                Profile
              </Button>
              <Button
                variant={activeView === 'matches' ? 'default' : 'ghost'}
                size="sm"
                className="h-8 rounded-full px-2 sm:px-3"
                onClick={() => setActiveView('matches')}
              >
                Matches
              </Button>
            </div>
            <Button variant="ghost" size="icon" onClick={handleShare} className="w-9 h-9">
              <Share2 className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onLogout} className="w-9 h-9" title="Logout">
              <LogOut className="w-4 h-4" />
            </Button>
            <Button asChild variant="ghost" size="sm" className="h-9 px-3">
              <Link href="/model-card">Model Card</Link>
            </Button>
            {activeView === 'profile' && (
              <Button
                variant="ghost"
                size="sm"
                className="h-9 px-3"
                onClick={() => setShowTrends((v) => !v)}
                disabled={!hasUploads}
              >
                {showTrends ? 'Profile' : 'Taste Trends'}
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-4 py-6 md:py-10 space-y-6">
        {/* Profile Hero */}
        <ProfileHero profile={profile} onRecommendationClick={onRecommendationClick} />

        {!hasUploads && (
          <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-4">
            <h3 className="text-lg font-semibold">Start Your Flavor Fingerprint</h3>
            <p className="text-sm text-muted-foreground">
              BiteMe analyzes each dish photo to estimate your long-term taste tendencies and short-term shifts.
              Upload your first meal to unlock trends, matches, and personalized insights.
            </p>
            <Button
              className="rounded-full"
              onClick={() => uploadRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
            >
              Upload your first dish
            </Button>
          </section>
        )}

        {activeView === 'matches' ? (
          <CompatibleUsers
            users={profile.compatible_users}
            currentUsername={profile.username}
            onInviteToEat={onInviteToEat}
          />
        ) : hasUploads && showTrends ? (
          <TasteTrends history={profile.taste_profile.history} />
        ) : hasUploads ? (
          <>
            {/* Two Column Layout on Desktop */}
            <div className="grid gap-6 lg:grid-cols-2">
              {/* Flavor Fingerprint */}
              <div className="lg:col-span-2">
                <FlavorFingerprint
                  dimensions={profile.taste_profile.dimensions}
                  refreshToken={fingerprintRefreshToken}
                />
              </div>

              {/* Taste Analysis */}
              <div className="lg:col-span-2">
                <TasteAnalysis profile={profile} />
              </div>
            </div>

            {/* Recent Uploads */}
            <RecentUploads uploads={profile.recent_uploads} onRemoveUpload={onRemoveUpload} />
            <ClickedRecommendations
              items={profile.clicked_recommendations}
              onRemove={onRemoveRecommendationClick}
            />
          </>
        ) : (
          <>
            <RecentUploads uploads={[]} onRemoveUpload={onRemoveUpload} />
            <ClickedRecommendations
              items={profile.clicked_recommendations}
              onRemove={onRemoveRecommendationClick}
            />
          </>
        )}

        {/* Upload Module */}
        <div ref={uploadRef}>
          <UploadModule onUpload={onUpload} isUploading={isUploading} />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 py-8 mt-12">
        <div className="max-w-3xl mx-auto px-4 text-center space-y-2">
          <p className="text-sm text-muted-foreground">
            Keep uploading to refine your taste profile
          </p>
          <p className="text-xs text-muted-foreground/60 italic">
            Because your dinner choices are absolutely saying something.
          </p>
        </div>
      </footer>
    </div>
  )
}
