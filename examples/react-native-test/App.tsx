import React, { useState, useCallback } from 'react';
import {
  SafeAreaView,
  ScrollView,
  TextInput,
  Text,
  TouchableOpacity,
  View,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import {
  LocalAI,
  getDeviceCapabilities,
  recommendQuantization,
  canRunModel,
} from 'local-llm-rn';
import type { LocalAIOptions } from 'local-llm-rn';

const MODEL_URL = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

export default function App() {
  const [status, setStatus] = useState('Press "Load Model" to start');
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [ai, setAi] = useState<LocalAI | null>(null);

  const showDeviceInfo = useCallback(() => {
    const caps = getDeviceCapabilities();
    const quant = recommendQuantization();
    setStatus(
      `RAM: ${(caps.totalRAM / 1e9).toFixed(1)} GB | ` +
      `GPU: ${caps.gpuName} (Metal ${caps.metalVersion}) | ` +
      `Recommended quant: ${quant}`
    );
  }, []);

  const loadModel = useCallback(async () => {
    setLoading(true);
    setStatus('Downloading model...');
    try {
      const instance = await LocalAI.create({
        model: MODEL_URL,
        compute: 'gpu',
        contextSize: 2048,
        onProgress: (pct) => setStatus(`Downloading: ${pct.toFixed(1)}%`),
      } as LocalAIOptions);
      setAi(instance);
      setStatus('Model loaded! Type a message below.');
    } catch (err: any) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const sendMessage = useCallback(async () => {
    if (!ai || !input.trim()) return;
    setLoading(true);
    setOutput('');
    try {
      const response = await ai.chat.completions.create({
        messages: [{ role: 'user', content: input }],
        stream: true,
      });
      let text = '';
      for await (const chunk of response) {
        const delta = chunk.choices[0]?.delta?.content ?? '';
        text += delta;
        setOutput(text);
      }
    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [ai, input]);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>local-llm-rn Test</Text>
      <Text style={styles.status}>{status}</Text>

      <View style={styles.buttons}>
        <TouchableOpacity style={styles.button} onPress={showDeviceInfo}>
          <Text style={styles.buttonText}>Device Info</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, ai && styles.buttonDisabled]}
          onPress={loadModel}
          disabled={!!ai || loading}
        >
          <Text style={styles.buttonText}>Load Model</Text>
        </TouchableOpacity>
      </View>

      <TextInput
        style={styles.input}
        placeholder="Type a message..."
        value={input}
        onChangeText={setInput}
        editable={!!ai && !loading}
      />
      <TouchableOpacity
        style={[styles.button, styles.sendButton]}
        onPress={sendMessage}
        disabled={!ai || loading}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Send</Text>
        )}
      </TouchableOpacity>

      <ScrollView style={styles.output}>
        <Text style={styles.outputText}>{output}</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: '#f5f5f5' },
  title: { fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginBottom: 8 },
  status: { fontSize: 12, color: '#666', textAlign: 'center', marginBottom: 16 },
  buttons: { flexDirection: 'row', gap: 8, marginBottom: 16 },
  button: {
    flex: 1, backgroundColor: '#007AFF', padding: 12,
    borderRadius: 8, alignItems: 'center',
  },
  buttonDisabled: { backgroundColor: '#999' },
  buttonText: { color: '#fff', fontWeight: '600' },
  sendButton: { marginBottom: 16 },
  input: {
    backgroundColor: '#fff', borderRadius: 8, padding: 12,
    fontSize: 16, marginBottom: 8, borderWidth: 1, borderColor: '#ddd',
  },
  output: { flex: 1, backgroundColor: '#fff', borderRadius: 8, padding: 12 },
  outputText: { fontSize: 14, lineHeight: 20 },
});
