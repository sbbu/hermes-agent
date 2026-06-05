import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'

const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  persistString: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  storedString: (key: string) => storage.get(key) ?? null
}))

const notifySpy = vi.fn()
const dismissSpy = vi.fn()

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => notifySpy(...args),
  dismissNotification: (...args: unknown[]) => dismissSpy(...args)
}))

const {
  $updateApply,
  $updateStatus,
  applyUpdates,
  checkUpdates,
  maybeNotifyUpdateAvailable,
  reportBackendContract,
  resetUpdateApplyState
} = await import('./updates')

const { $connection } = await import('./session')

const status = (over: Partial<DesktopUpdateStatus> = {}): DesktopUpdateStatus => ({
  supported: true,
  behind: 3,
  targetSha: 'sha-a',
  fetchedAt: 0,
  ...over
})

const lastToast = () => notifySpy.mock.calls.at(-1)?.[0] as { onDismiss: () => void }

describe('maybeNotifyUpdateAvailable', () => {
  beforeEach(() => {
    storage.clear()
    notifySpy.mockClear()
    dismissSpy.mockClear()
    $connection.set(null)
    $updateStatus.set(null)
    resetUpdateApplyState()
    vi.useRealTimers()
  })

  it('shows when an update is available and not snoozed', () => {
    maybeNotifyUpdateAvailable(status())
    expect(notifySpy).toHaveBeenCalledTimes(1)
  })

  it('stays quiet for new commits once the toast was closed', () => {
    maybeNotifyUpdateAvailable(status())
    lastToast().onDismiss() // user closes it → cooldown starts
    notifySpy.mockClear()

    // A different commit lands while still within the cooldown window.
    maybeNotifyUpdateAvailable(status({ targetSha: 'sha-b', behind: 9 }))
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('re-shows once the cooldown elapses', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)

    maybeNotifyUpdateAvailable(status())
    lastToast().onDismiss()
    notifySpy.mockClear()

    vi.setSystemTime(25 * 60 * 60 * 1000) // > 24h cooldown
    maybeNotifyUpdateAvailable(status({ targetSha: 'sha-b' }))
    expect(notifySpy).toHaveBeenCalledTimes(1)
  })

  it('does nothing when already up to date', () => {
    maybeNotifyUpdateAvailable(status({ behind: 0 }))
    expect(notifySpy).not.toHaveBeenCalled()
  })

  it('does not call the update bridge while connected to a remote backend', async () => {
    const bridgeCheck = vi.fn(async () => status())
    window.hermesDesktop = { updates: { check: bridgeCheck } } as never
    $connection.set({ mode: 'remote' } as never)

    const result = await checkUpdates()

    expect(bridgeCheck).not.toHaveBeenCalled()
    expect(result?.supported).toBe(false)
    expect(result?.reason).toBe('remote-connection')
    expect(dismissSpy).toHaveBeenCalledWith('desktop-update-available')
  })

  it('refuses to apply updates while connected to a remote backend', async () => {
    const bridgeApply = vi.fn(async () => ({ ok: true }))
    window.hermesDesktop = { updates: { apply: bridgeApply } } as never
    $connection.set({ mode: 'remote' } as never)

    const result = await applyUpdates()

    expect(bridgeApply).not.toHaveBeenCalled()
    expect(result.ok).toBe(false)
    expect(result.error).toBe('remote-connection')
    expect($updateApply.get().stage).toBe('error')
  })

  it('surfaces remote backend skew without offering in-app self-update', () => {
    $connection.set({ mode: 'remote' } as never)

    reportBackendContract(0)

    const notification = notifySpy.mock.calls.at(-1)?.[0]
    expect(notification.title).toBe('Remote backend out of date')
    expect(notification.action.label).toBe('Details')
    expect(notification.message).toContain('Update the remote host')
  })
})
