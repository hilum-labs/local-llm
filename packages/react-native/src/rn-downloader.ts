import NativeLocalLLM from './NativeLocalLLM';
import { NativeEventEmitter, NativeModules } from 'react-native';
import type { DownloadAdapter } from './download-adapter';

const emitter = new NativeEventEmitter(NativeModules.LocalLLM);

export function createNativeDownloader(): DownloadAdapter {
  return {
    download(url, destPath, onProgress) {
      return new Promise((resolve, reject) => {
        const subs = [
          emitter.addListener('onDownloadProgress', (e) => {
            if (e.url !== url) return;
            onProgress?.(e.downloaded, e.total, e.percent);
          }),
          emitter.addListener('onDownloadComplete', (e) => {
            if (e.url !== url) return;
            subs.forEach((s) => s.remove());
            resolve();
          }),
          emitter.addListener('onDownloadError', (e) => {
            if (e.url !== url) return;
            subs.forEach((s) => s.remove());
            reject(new Error(e.error));
          }),
        ];
        NativeLocalLLM.downloadModel(url, destPath);
      });
    },

    cancel(url) {
      NativeLocalLLM.cancelDownload(url);
    },

    supportsResume: false,
  };
}
