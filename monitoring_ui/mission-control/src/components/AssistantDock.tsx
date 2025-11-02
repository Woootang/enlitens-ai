import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  Fab,
  IconButton,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { useAssistant } from '../hooks/useAssistant';
import { useDashboardStore } from '../state/useDashboardStore';

const ASSISTANT_VISIBLE_KEY = 'mission-control-assistant-open';

const loadInitialOpen = () => {
  if (typeof window === 'undefined') return false;
  return window.localStorage.getItem(ASSISTANT_VISIBLE_KEY) === 'true';
};

const persistOpen = (value: boolean) => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(ASSISTANT_VISIBLE_KEY, value ? 'true' : 'false');
};

const escapeHtml = (value: string) =>
  value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');

const renderMarkdown = (value: string) => {
  const escaped = escapeHtml(value);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br />');
};

export const AssistantDock = () => {
  const [open, setOpen] = useState<boolean>(loadInitialOpen);
  const [draft, setDraft] = useState('');
  const [pendingAction, setPendingAction] = useState<ReturnType<typeof useAssistant>['actions'][number] | null>(null);
  const { messages, pending, error, sendPrompt, actions, executeAction } = useAssistant();
  const insights = useDashboardStore((state) => state.insights);

  useEffect(() => {
    persistOpen(open);
  }, [open]);

  const handleSend = () => {
    if (!draft.trim()) return;
    void sendPrompt(draft.trim(), false);
    setDraft('');
  };

  const insightBadges = useMemo(() => insights.slice(0, 3), [insights]);

  return (
    <>
      <Fab
        color={open ? 'secondary' : 'primary'}
        aria-label="assistant"
        onClick={() => setOpen((value) => !value)}
        sx={{ position: 'fixed', bottom: 32, right: open ? 360 + 32 : 32, zIndex: 1300 }}
      >
        <ChatIcon />
      </Fab>
      {open && (
        <Paper
          elevation={16}
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            width: { xs: 'calc(100% - 48px)', sm: 360 },
            maxHeight: 520,
            display: 'flex',
            flexDirection: 'column',
            borderRadius: 3,
            border: '1px solid rgba(99,179,237,0.4)',
            background: 'rgba(13, 20, 38, 0.92)',
            zIndex: 1299,
          }}
        >
          <Box px={2} py={1.5} display="flex" alignItems="center" justifyContent="space-between">
            <Stack spacing={0.25}>
              <Typography variant="subtitle1" fontWeight={600}>
                Mission Control Assistant
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Ask about system health or trigger automation
              </Typography>
            </Stack>
            <Tooltip title="Hide assistant">
              <IconButton size="small" onClick={() => setOpen(false)}>
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          <Divider sx={{ borderColor: 'rgba(255,255,255,0.06)' }} />
          {insightBadges.length > 0 && (
            <Stack direction="row" spacing={1} px={2} py={1} flexWrap="wrap">
              {insightBadges.map((item, index) => (
                <Chip key={`${item}-${index}`} label={item} size="small" color="info" variant="filled" />
              ))}
            </Stack>
          )}
          <Box flex={1} px={2} py={1.5} overflow="auto">
            <Stack spacing={1.5}>
              {messages.map((message) => (
                <Box
                  key={message.id}
                  sx={{
                    alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                    backgroundColor: message.role === 'user' ? 'rgba(59,130,246,0.28)' : 'rgba(255,255,255,0.05)',
                    borderRadius: 2,
                    px: 1.5,
                    py: 1,
                    maxWidth: '90%',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  <Typography variant="caption" color="text.secondary" display="block" mb={0.5}>
                    {message.role === 'user' ? 'You' : message.auto ? 'Assistant (auto)' : 'Assistant'}
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.primary"
                    whiteSpace="pre-wrap"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
                  />
                </Box>
              ))}
              {messages.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  Start a conversation — ask about current anomalies, agent performance, or type “help” to see suggestions.
                </Typography>
              )}
            </Stack>
          </Box>
          {error && (
            <Alert severity="warning" sx={{ mx: 2, mb: 1 }}>
              {error}
            </Alert>
          )}
          <Divider sx={{ borderColor: 'rgba(255,255,255,0.06)' }} />
          <Stack direction="row" spacing={1} px={2} py={1.5} alignItems="center">
            <TextField
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  handleSend();
                }
              }}
              placeholder={pending ? 'Waiting for assistant…' : 'Ask a question'}
              fullWidth
              size="small"
              multiline
              minRows={1}
              maxRows={3}
              disabled={pending}
            />
            <IconButton color="primary" aria-label="send message" disabled={pending || !draft.trim()} onClick={handleSend}>
              <SendIcon fontSize="small" />
            </IconButton>
          </Stack>
          <Divider sx={{ borderColor: 'rgba(255,255,255,0.06)' }} />
          <Stack direction="row" spacing={1} px={2} py={1} flexWrap="wrap">
            {actions.map((action) => (
              <Button
                key={action.id}
                size="small"
                variant="outlined"
                color="info"
                startIcon={<PlayArrowIcon fontSize="small" />}
                onClick={() => setPendingAction(action)}
              >
                {action.label}
              </Button>
            ))}
          </Stack>
        </Paper>
      )}
      <Dialog open={Boolean(pendingAction)} onClose={() => setPendingAction(null)}>
        <DialogTitle>Confirm assistant action</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {pendingAction?.confirm}
            {pendingAction?.description ? ` (${pendingAction.description})` : ''}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPendingAction(null)}>Cancel</Button>
          <Button
            onClick={() => {
              if (!pendingAction) return;
              void executeAction(pendingAction);
              setPendingAction(null);
            }}
            color="primary"
            autoFocus
          >
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

