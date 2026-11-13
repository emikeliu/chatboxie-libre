import type { BuiltInBrowserTabs } from './built-in-browser'

export interface ElectronIPC {
  invoke: (channel: string, ...args: any[]) => Promise<any>
  getPathForFile: (file: File) => string
  onSystemThemeChange: (callback: () => void) => () => void
  onWindowMaximizedChanged: (callback: (_: Electron.IpcRendererEvent, windowMaximized: boolean) => void) => () => void
  onTaskCloseRequest: (callback: () => void) => () => void
  onWindowShow: (callback: () => void) => () => void
  onWindowFocused: (callback: () => void) => () => void
  addMcpStdioTransportEventListener: <Args extends unknown[]>(
    transportId: string,
    event: string,
    callback?: (...args: Args) => void
  ) => () => void
  onNavigate: (callback: (path: string) => void) => () => void
  // 内置 skill 后台同步完成（有更新）时由 main 推送，renderer 据此刷新 skill 列表与工具缓存
  onSkillsBuiltinUpdated: (callback: () => void) => () => void
  onBuiltInBrowserTabs: (callback: (tabs: BuiltInBrowserTabs) => void) => () => void
}
