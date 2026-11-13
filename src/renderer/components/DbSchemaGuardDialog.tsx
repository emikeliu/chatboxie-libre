import { Button, Modal, Stack, Text } from '@mantine/core'
import { useTranslation } from 'react-i18next'
import { buildChatboxUrl } from '@/packages/remote'
import platform from '@/platform'
import { useDbSchemaGuardStore } from '@/storage/db-schema-guard'

/**
 * Blocking guidance for the two IndexedDB schema-mismatch situations
 * (see docs/technical/storage.md, "IndexedDB 版本策略"):
 * - schema-too-new: the user downgraded across a schema bump; data cannot be
 *   read by this build but is intact — guide them to update from the download page.
 * - upgraded-elsewhere: another window/tab bumped the schema while this one
 *   held a connection; a reload picks up the new state.
 */
export default function DbSchemaGuardDialog() {
  const { t, i18n } = useTranslation()
  const schemaTooNewDbName = useDbSchemaGuardStore((s) => s.schemaTooNewDbName)
  const upgradedElsewhereDbName = useDbSchemaGuardStore((s) => s.upgradedElsewhereDbName)
  const upgradeBlockedDbName = useDbSchemaGuardStore((s) => s.upgradeBlockedDbName)

  if (schemaTooNewDbName) {
    return (
      <Modal
        opened
        onClose={() => {}}
        withCloseButton={false}
        closeOnClickOutside={false}
        closeOnEscape={false}
        centered
        title={t('Your data was created by a newer version of Chatbox')}
      >
        <Stack gap="sm">
          <Text size="sm">
            {t(
              'This version of Chatbox cannot read data created by a newer version. Your data is safe — update Chatbox to the latest version to continue.'
            )}
          </Text>
          <UpdateAction language={i18n.language} />
        </Stack>
      </Modal>
    )
  }

  if (upgradeBlockedDbName) {
    // Clears automatically once the other window releases its connection and open succeeds.
    return (
      <Modal
        opened
        onClose={() => {}}
        withCloseButton={false}
        closeOnClickOutside={false}
        closeOnEscape={false}
        centered
        title={t('Waiting for other Chatbox windows')}
      >
        <Text size="sm">{t('Close other Chatbox windows or tabs to finish updating the local database.')}</Text>
      </Modal>
    )
  }

  if (upgradedElsewhereDbName) {
    return (
      <Modal
        opened
        onClose={() => {}}
        withCloseButton={false}
        closeOnClickOutside={false}
        closeOnEscape={false}
        centered
        title={t('Chatbox was updated in another window')}
      >
        <Stack gap="sm">
          <Text size="sm">{t('Reload this page to continue.')}</Text>
          <Button onClick={() => window.location.reload()}>{t('Reload')}</Button>
        </Stack>
      </Modal>
    )
  }

  return null
}

function UpdateAction({ language }: { language: string }) {
  const { t } = useTranslation()

  // Auto-update has been removed from this edition: reload on web and point
  // every other platform at the download page.
  return (
    <Stack gap="xs">
      <Button onClick={() => window.location.reload()}>{t('Reload')}</Button>
      <Button
        variant="default"
        onClick={() => platform.openLink(buildChatboxUrl(`/redirect_app/check_update/${language}`))}
      >
        {t('Check Update')}
      </Button>
    </Stack>
  )
}
