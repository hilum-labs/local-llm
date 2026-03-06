export interface DownloadAdapter {
  download(
    url: string,
    destPath: string,
    onProgress?: (downloaded: number, total: number, percent: number) => void,
  ): Promise<void>;

  cancel?(url: string): void;

  supportsResume?: boolean;
  resume?(
    url: string,
    destPath: string,
    resumeToken: unknown,
    onProgress?: (downloaded: number, total: number, percent: number) => void,
  ): Promise<void>;
}
