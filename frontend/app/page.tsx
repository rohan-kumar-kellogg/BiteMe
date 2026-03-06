'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
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
  const [loadingReason, setLoadingReason] = useState<'bootstrap' | 'submit' | ''>('')
  const SESSION_KEY = 'biteme_session_v1'
  const authFlowVersionRef = useRef(0)
  const restoreStartedRef = useRef(false)
  const activeRequestsRef = useRef({
    auth: 0,
    profileRefresh: 0,
    upload: 0,
  })

  const transitionAppState = useCallback((nextState: AppState, reason: string) => {
    setAppState((prevState) => {
      console.log('[appState]', prevState, '->', nextState, reason)
      return nextState
    })
  }, [])

  const renderLoadingState = useCallback((reason: string) => {
    const sessionExists =
      typeof window !== 'undefined' ? Boolean(localStorage.getItem(SESSION_KEY)) : false
    const route = typeof window !== 'undefined' ? window.location.pathname : 'server'
    console.log('[loading render]', {
      reason,
      appState,
      profileIsNull: profile === null,
      sessionExists,
      isLoading,
      isUploading,
      loadingReason,
      route,
      activeRequests: activeRequestsRef.current,
    })
    return (
      <LoadingState
        title={loadingReason === 'submit' ? 'Loading your profile...' : 'Restoring your session...'}
        subtitle={
          loadingReason === 'submit'
            ? 'Fetching your latest flavor fingerprint'
            : 'Loading your saved BiteMe profile'
        }
      />
    )
  }, [SESSION_KEY, appState, profile, isLoading, isUploading, loadingReason])

  const refreshProfile = useCallback(async (username: string) => {
    activeRequestsRef.current.profileRefresh += 1
    console.info('[profile] refresh started', { username })
    try {
      const [profileData, compatibleUsers] = await Promise.all([
        getUserProfile(username),
        getCompatibleUsers(username),
      ])
      console.info('[profile] set profile from getUserProfile success', {
        username,
        hasProfile: Boolean(profileData),
      })
      setProfile({
        ...profileData,
        compatible_users: compatibleUsers,
      })
      console.info('[profile] refresh success', {
        username,
        upload_count: profileData.total_uploads,
      })
    } catch (error) {
      console.error('[profile] refresh failed', error)
      throw error
    } finally {
      activeRequestsRef.current.profileRefresh = Math.max(0, activeRequestsRef.current.profileRefresh - 1)
    }
  }, [])

  const handleUsernameSubmit = useCallback(async (identity: SessionIdentity) => {
    const flowVersion = ++authFlowVersionRef.current
    activeRequestsRef.current.auth += 1
    console.info('[auth] submit start', { flowVersion, username: identity.username })
    setLoadingReason('submit')
    setIsLoading(true)
    transitionAppState('loading', 'username submit started')
    setErrorMessage('')

    try {
      const result = await loadOrCreateUser(identity.username, identity.email)
      if (flowVersion !== authFlowVersionRef.current) {
        console.warn('[auth] submit stale success ignored', { flowVersion, current: authFlowVersionRef.current })
        return
      }
      setActiveUsername(result.user.username)
      setActiveEmail(result.user.email || identity.email)
      localStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ username: result.user.username, email: result.user.email || identity.email })
      )
      setIsNewUser(result.isNew)
      await refreshProfile(result.user.username)
      if (flowVersion !== authFlowVersionRef.current) {
        console.warn('[auth] submit stale post-refresh ignored', { flowVersion, current: authFlowVersionRef.current })
        return
      }
      transitionAppState('profile', 'username submit success')
    } catch (error) {
      if (flowVersion !== authFlowVersionRef.current) {
        console.warn('[auth] submit stale failure ignored', { flowVersion, current: authFlowVersionRef.current })
        return
      }
      setErrorMessage('Failed to load profile. Please try again.')
      transitionAppState('error', 'username submit failed')
    } finally {
      if (flowVersion === authFlowVersionRef.current) {
        setIsLoading(false)
        setLoadingReason('')
      }
      activeRequestsRef.current.auth = Math.max(0, activeRequestsRef.current.auth - 1)
      console.info('[auth] submit finished', { flowVersion })
    }
  }, [refreshProfile, transitionAppState])

  const handleUpload = useCallback(async (file: File) => {
    if (!profile && !activeUsername) {
      throw new Error('No active user session for upload.')
    }

    const username = profile?.username || activeUsername
    activeRequestsRef.current.upload += 1
    console.info('[upload] started', {
      username,
      file: file?.name || 'unknown',
      size: file?.size || 0,
    })
    setIsUploading(true)

    try {
      console.info('[upload] request sent', { username })
      const uploadResponse = await uploadUserImage(username, file)
      console.info('[upload] request success', {
        username,
        predicted_label: uploadResponse?.prediction?.predicted_label,
        abstained: uploadResponse?.prediction?.abstained,
        rejected_as_not_food: uploadResponse?.prediction?.rejected_as_not_food,
      })
      await refreshProfile(username)
      localStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ username, email: profile?.email || activeEmail })
      )
      if (isNewUser) {
        setIsNewUser(false)
      }
      transitionAppState('profile', 'upload + refresh success')
    } catch (error) {
      console.error('[upload] request failed', error)
      const message = error instanceof Error ? error.message : 'Upload failed. Please try again.'
      throw new Error(message)
    } finally {
      console.info('[ui] clearing analyzing state')
      setIsUploading(false)
      activeRequestsRef.current.upload = Math.max(0, activeRequestsRef.current.upload - 1)
    }
  }, [profile, activeUsername, refreshProfile, isNewUser, activeEmail, transitionAppState])

  const handleBack = useCallback(() => {
    transitionAppState('welcome', 'back pressed')
    setErrorMessage('')
  }, [transitionAppState])

  const handleLogout = useCallback(() => {
    localStorage.removeItem(SESSION_KEY)
    console.info('[profile] cleared on logout')
    setProfile(null)
    setActiveUsername('')
    setActiveEmail('')
    authFlowVersionRef.current += 1
    transitionAppState('welcome', 'logout')
    setErrorMessage('')
    setLoadingReason('')
    setIsLoading(false)
    setIsUploading(false)
  }, [transitionAppState])

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
    transitionAppState('welcome', 'error retry')
    setErrorMessage('')
  }, [transitionAppState])

  useEffect(() => {
    if (restoreStartedRef.current) return
    restoreStartedRef.current = true
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
      console.info('[session] restore start', { hasSavedSession: Boolean(saved) })
      if (!saved) {
        console.info('[session] restore skipped (no saved session)')
        return
      }
      const flowVersion = ++authFlowVersionRef.current
      activeRequestsRef.current.auth += 1
      setLoadingReason('bootstrap')
      transitionAppState('loading', 'session restore start')
      setIsLoading(true)
      try {
        console.info('[session] restore request start', { username: saved.username, flowVersion })
        setActiveUsername(saved.username)
        setActiveEmail(saved.email || '')
        await refreshProfile(saved.username)
        if (flowVersion !== authFlowVersionRef.current) {
          console.warn('[session] restore stale success ignored', {
            flowVersion,
            current: authFlowVersionRef.current,
          })
          return
        }
        setIsNewUser(false)
        console.info('[session] restore success', { username: saved.username, flowVersion })
        transitionAppState('profile', 'session restore success')
      } catch (error) {
        if (flowVersion !== authFlowVersionRef.current) {
          console.warn('[session] restore stale failure ignored', {
            flowVersion,
            current: authFlowVersionRef.current,
          })
          return
        }
        console.error('[session] restore failed', error)
        localStorage.removeItem(SESSION_KEY)
        transitionAppState('welcome', 'session restore failed')
      } finally {
        if (flowVersion === authFlowVersionRef.current) {
          setIsLoading(false)
          setLoadingReason('')
        }
        activeRequestsRef.current.auth = Math.max(0, activeRequestsRef.current.auth - 1)
        console.info('[session] restore finished', { flowVersion })
      }
    }
    void restore()
  }, [refreshProfile, transitionAppState])

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
      return renderLoadingState('appState=loading')

    case 'new-user':
      return profile ? (
        <NewUserState 
          username={profile.username}
          onUpload={handleUpload}
          isUploading={isUploading}
          onBack={handleBack}
        />
      ) : (
        <ErrorState
          message="We couldn't load your profile."
          onGoHome={() => transitionAppState('welcome', 'new-user state missing profile')}
          goHomeLabel="Back to start"
        />
      )

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
      ) : (
        <ErrorState
          message="We couldn't load your profile."
          onGoHome={() => transitionAppState('welcome', 'profile state missing profile')}
          goHomeLabel="Back to start"
        />
      )

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
