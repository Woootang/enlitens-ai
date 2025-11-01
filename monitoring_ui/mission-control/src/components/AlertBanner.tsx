import { Alert, AlertTitle, Typography } from '@mui/material';
import { SeverityLevel } from '../types';

const severityMap: Record<SeverityLevel, { severity: 'info' | 'warning' | 'error'; title: string }> = {
  normal: { severity: 'info', title: 'Nominal' },
  warning: { severity: 'warning', title: 'Attention' },
  critical: { severity: 'error', title: 'Action Required' },
};

interface AlertBannerProps {
  severity: SeverityLevel;
  messages: string[];
  connectionStatus: string;
}

export const AlertBanner = ({ severity, messages, connectionStatus }: AlertBannerProps) => {
  const config = severityMap[severity];
  const message = messages[0] ?? 'System stable and awaiting new tasks.';

  return (
    <Alert severity={config.severity} variant="outlined" sx={{ borderRadius: 3, borderColor: 'primary.main' }}>
      <AlertTitle sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
        <Typography component="span" fontWeight={700} textTransform="uppercase" fontSize={12} letterSpacing={2} color="primary.light">
          {config.title}
        </Typography>
        <Typography component="span" fontSize={12} color="text.secondary">
          Link: {connectionStatus}
        </Typography>
      </AlertTitle>
      <Typography fontSize={14}>{message}</Typography>
    </Alert>
  );
};
