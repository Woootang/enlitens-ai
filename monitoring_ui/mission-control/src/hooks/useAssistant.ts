import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { composeAssistantContext } from '../services/context';
import { postAction, postChatMessage } from '../services/api';
import { useDashboardStore } from '../state/useDashboardStore';

export type ChatRole = 'user' | 'assistant' | 'system' | 'auto';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  auto?: boolean;
}

interface AssistantAction {
  id: string;
  label: string;
  description: string;
  endpoint: string;
  payload?: Record<string, unknown>;
  confirm: string;
}

const ACTION_REGISTRY: AssistantAction[] = [
  {
    id: 'retry-document',
    label: 'Retry active document',
    description: 'Re-enqueue the current document for processing.',
    endpoint: '/api/actions/retry-document',
    payload: {},
    confirm: 'Retrying will re-run the current document with the existing pipeline. Continue?',
  },
  {
    id: 'toggle-verbose',
    label: 'Toggle verbose logging',
    description: 'Enable or disable verbose logging for the supervisor.',
    endpoint: '/api/actions/toggle-verbose',
    payload: {},
    confirm: 'Toggle verbose logging for the orchestration? This may affect performance.',
  },
];

const AUTO_PROMPT_INTERVAL = 30_000;

const buildMessage = (role: ChatRole, content: string, auto = false): ChatMessage => ({
  id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  role,
  content,
  createdAt: new Date().toISOString(),
  auto,
});

export const useAssistant = () => {
  const summary = useDashboardStore((state) => state.summary);
  const agents = useDashboardStore((state) => state.agents);
  const quality = useDashboardStore((state) => state.quality);
  const insights = useDashboardStore((state) => state.insights);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const autoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastAutoAtRef = useRef<number>(0);

  const contextPayload = useMemo(
    () => composeAssistantContext({ summary, agents, quality, insights }),
    [summary, agents, quality, insights],
  );

  const pushMessage = useCallback((message: ChatMessage) => {
    setMessages((current) => [...current, message]);
  }, []);

  const sendPrompt = useCallback(
    async (prompt: string, auto = false) => {
      if (!prompt.trim()) return;
      const userMessage = buildMessage(auto ? 'auto' : 'user', prompt, auto);
      pushMessage(userMessage);
      setPending(true);
      setError(null);
      try {
        const response = await postChatMessage({ message: prompt, context: contextPayload });
        if (response) {
          pushMessage(buildMessage('assistant', response, auto));
        } else {
          setError('Assistant did not respond. Check connectivity.');
        }
      } catch (err) {
        console.error(err);
        setError('Failed to contact assistant service.');
      } finally {
        setPending(false);
      }
    },
    [contextPayload, pushMessage],
  );

  const scheduleAutoPrompt = useCallback(() => {
    if (autoTimerRef.current) {
      clearInterval(autoTimerRef.current);
    }
    autoTimerRef.current = setInterval(() => {
      const now = Date.now();
      if (now - lastAutoAtRef.current < AUTO_PROMPT_INTERVAL) {
        return;
      }
      lastAutoAtRef.current = now;
      void sendPrompt('Analyze the latest telemetry for anomalies and summarize key findings.', true);
    }, AUTO_PROMPT_INTERVAL);
  }, [sendPrompt]);

  useEffect(() => {
    scheduleAutoPrompt();
    return () => {
      if (autoTimerRef.current) {
        clearInterval(autoTimerRef.current);
      }
    };
  }, [scheduleAutoPrompt]);

  const executeAction = useCallback(async (action: AssistantAction) => {
    const success = await postAction(action.endpoint, action.payload ?? {});
    pushMessage(
      buildMessage(
        'assistant',
        success
          ? `✅ Action “${action.label}” executed successfully.`
          : `⚠️ Failed to execute action “${action.label}”. Check backend logs for details.`,
      ),
    );
    return success;
  }, [pushMessage]);

  return {
    messages,
    pending,
    error,
    sendPrompt,
    actions: ACTION_REGISTRY,
    executeAction,
  };
};

