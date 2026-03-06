'use client'

import { useState, useCallback, useEffect } from 'react'
import { WelcomeScreen } from '@/components/welcome-screen'
import { ProfilePage } from '@/components/profile-page'
import { NewUserState } from '@/components/new-user-state'
import { LoadingState } from '@/components/loading-state'
import { ErrorState } from '@/components/error-state'
import type { AppState, SessionIdentity, UserProfile } from '@/lib/types'
import {
  createInvite,
  getCompatibleUsers,
  getUserProfile,
  loadOrCreateUser,
  removeRecommendationFeedback,
  removeUserUpload,
  sendRecommendationFeedback,
  uploadUserImage,
} from '@/lib/api'

export default function Home() {
  const [appState, setAppState] = useState<AppState>('welcome')
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [isNewUser, setIsNewUser] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [activeUsername, setActiveUsername] = useState<string>('')
  const [activeEmail, setActiveEmail] = useState<string>('')
  const [fingerprintRefreshToken, setFingerprintRefreshToken] = useState(0)
  const SESSION_KEY = 'biteme_session_v1'

  const refreshProfile = useCallback(async (username: string) => {
    const [profileData, compatibleUsers] = await Promise.all([
      getUserProfile(username),
      getCompatibleUsers(username),
    ])
    setProfile({
      ...profileData,
      compatible_users: compatibleUsers,
    })
  }, [])

  const handleUsernameSubmit = useCallback(async (identity: SessionIdentity) => {
    setIsLoading(true)
    setAppState('loading')
    setErrorMessage('')

    try {
      const result = await loadOrCreateUser(identity.username, identity.email)
      setActiveUsername(result.user.username)
      setActiveEmail(result.user.email || identity.email)
      localStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ username: result.user.username, email: result.user.email || identity.email })
      )
      setIsNewUser(result.isNew)
      await refreshProfile(result.user.username)
      setAppState('profile')
    } catch (error) {
      setErrorMessage('Failed to load profile. Please try again.')
      setAppState('error')
    } finally {
      setIsLoading(false)
    }
  }, [refreshProfile])

  const handleUpload = useCallback(async (file: File) => {
    if (!profile && !activeUsername) return

    setIsUploading(true)

    try {
      const username = profile?.username || activeUsername
      await uploadUserImage(username, file)
      await refreshProfile(username)
      localStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ username, email: profile?.email || activeEmail })
      )
      if (isNewUser) {
        setIsNewUser(false)
      }
      setAppState('profile')
    } catch (error) {
      throw error
    } finally {
      setIsUploading(false)
    }
  }, [profile, activeUsername, refreshProfile, isNewUser, activeEmail])

  const handleBack = useCallback(() => {
    setAppState('welcome')
    setErrorMessage('')
  }, [])

  const handleLogout = useCallback(() => {
    localStorage.removeItem(SESSION_KEY)
    setProfile(null)
    setActiveUsername('')
    setActiveEmail('')
    setAppState('welcome')
    setErrorMessage('')
  }, [])

  const handleRecommendationClick = useCallback(async (dishLabel: string, cuisine = '') => {
    const username = profile?.username || activeUsername
    if (!username) return
    await sendRecommendationFeedback(username, dishLabel, cuisine)
    await refreshProfile(username)
    setFingerprintRefreshToken(Date.now())
  }, [profile, activeUsername, refreshProfile])

  const handleRemoveRecommendationClick = useCallback(async (eventId: string) => {
    const username = profile?.username || activeUsername
    if (!username) return
    await removeRecommendationFeedback(username, eventId)
    await refreshProfile(username)
  }, [profile, activeUsername, refreshProfile])

  const handleRemoveUpload = useCallback(async (uploadId: string) => {
    const username = profile?.username || activeUsername
    if (!username) return
    await removeUserUpload(username, uploadId)
    await refreshProfile(username)
  }, [profile, activeUsername, refreshProfile])

  const handleRetry = useCallback(() => {
    setAppState('welcome')
    setErrorMessage('')
  }, [])

  useEffect(() => {
    if (appState !== 'welcome') return
    let alive = true
    const parseSession = (raw: string | null): SessionIdentity | null => {
      if (!raw) return null
      try {
        const parsed = JSON.parse(raw)
        if (parsed && typeof parsed.username === 'string' && typeof parsed.email === 'string') {
          return { username: parsed.username, email: parsed.email }
        }
      } catch {
        // legacy session format fallback
        if (typeof raw === 'string' && raw.trim().length > 1) {
          return { username: raw.trim(), email: '' }
        }
      }
      return null
    }
    const restore = async () => {
      const saved = parseSession(localStorage.getItem(SESSION_KEY))
      if (!saved) return
      setAppState('loading')
      setIsLoading(true)
      try {
        if (!alive) return
        setActiveUsername(saved.username)
        setActiveEmail(saved.email || '')
        await refreshProfile(saved.username)
        if (!alive) return
        setIsNewUser(false)
        setAppState('profile')
      } catch {
        localStorage.removeItem(SESSION_KEY)
        if (!alive) return
        setAppState('welcome')
      } finally {
        if (alive) setIsLoading(false)
      }
    }
    void restore()
    return () => {
      alive = false
    }
  }, [refreshProfile, appState])

  const handleInviteToEat = useCallback(
    async (payload: { to_username: string; to_email?: string; restaurant_name?: string; date?: string; time?: string; message?: string }) => {
      const username = profile?.username || activeUsername
      if (!username) return
      await createInvite(username, payload)
    },
    [profile, activeUsername]
  )

  // Render based on app state
  switch (appState) {
    case 'welcome':
      return (
        <WelcomeScreen 
          onSubmit={handleUsernameSubmit} 
          isLoading={isLoading}
          initialUsername={activeUsername}
          initialEmail={activeEmail}
        />
      )

    case 'loading':
      return <LoadingState />

    case 'new-user':
      return profile ? (
        <NewUserState 
          username={profile.username}
          onUpload={handleUpload}
          isUploading={isUploading}
          onBack={handleBack}
        />
      ) : <LoadingState />

    case 'profile':
      return profile ? (
        <ProfilePage 
          profile={profile}
          fingerprintRefreshToken={fingerprintRefreshToken}
          onBack={handleBack}
          onLogout={handleLogout}
          onUpload={handleUpload}
          onInviteToEat={handleInviteToEat}
          onRecommendationClick={handleRecommendationClick}
          onRemoveRecommendationClick={handleRemoveRecommendationClick}
          onRemoveUpload={handleRemoveUpload}
          isUploading={isUploading}
        />
      ) : <LoadingState />

    case 'error':
      return (
        <ErrorState 
          message={errorMessage}
          onRetry={handleRetry}
          onGoHome={handleBack}
        />
      )

    default:
      return (
        <WelcomeScreen
          onSubmit={handleUsernameSubmit}
          isLoading={isLoading}
          initialUsername={activeUsername}
          initialEmail={activeEmail}
        />
      )
  }
}
