import { Card, CardContent, List, ListItem, ListItemAvatar, Avatar, ListItemText, Typography, Tooltip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import PendingIcon from '@mui/icons-material/Pending';
import { PlanStep } from '../types';
import { useDashboardStore } from '../state/useDashboardStore';

const iconMap = (status: PlanStep['status']) => {
  switch (status) {
    case 'completed':
      return { icon: CheckCircleIcon, color: 'success.main', label: 'Completed' };
    case 'in_progress':
      return { icon: AutorenewIcon, color: 'warning.main', label: 'In Progress' };
    case 'failed':
      return { icon: WarningAmberIcon, color: 'error.main', label: 'Failed' };
    case 'skipped':
      return { icon: PendingIcon, color: 'info.main', label: 'Skipped' };
    default:
      return { icon: PendingIcon, color: 'text.secondary', label: 'Pending' };
  }
};

type PlanPanelProps = {
  steps: PlanStep[];
  visible: boolean;
};

export const PlanPanel = ({ steps, visible }: PlanPanelProps) => {
  const highlightAgentId = useDashboardStore((state) => state.highlightAgentId);
  const highlightPlanStepId = useDashboardStore((state) => state.highlightPlanStepId);
  const actions = useDashboardStore((state) => state.actions);

  if (!visible) {
    return (
      <Card sx={{ border: '1px dashed rgba(255,255,255,0.3)', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CardContent>
          <Typography align="center" color="text.secondary">
            Plan view collapsed — toggle from the summary panel to monitor orchestration.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', border: '1px solid rgba(255,255,255,0.2)' }}>
      <CardContent sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
        {steps.length === 0 ? (
          <Typography align="center" color="text.secondary" sx={{ py: 8 }}>
            Waiting for supervisor plan…
          </Typography>
        ) : (
          <List dense sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {steps.map((step, index) => {
              const { icon: IconComponent, color, label } = iconMap(step.status);
              const isLinked = step.relatedAgentId && step.relatedAgentId === highlightAgentId;
              const isFocused = step.id === highlightPlanStepId;
              return (
                <Tooltip
                  key={step.id}
                  title={step.description ?? undefined}
                  placement="right"
                  arrow
                  enterDelay={200}
                >
                  <ListItem
                    onMouseEnter={() => actions.setHighlightPlanStep(step.id, step.relatedAgentId)}
                    onMouseLeave={() => actions.setHighlightPlanStep(undefined, undefined)}
                    onClick={() => actions.setHighlightPlanStep(step.id, step.relatedAgentId)}
                    sx={{
                      borderRadius: 2,
                      border:
                        isFocused || isLinked
                          ? '1px solid rgba(99,179,237,0.6)'
                          : '1px solid rgba(255,255,255,0.08)',
                      backgroundColor:
                        isFocused || isLinked ? 'rgba(99, 179, 237, 0.18)' : 'rgba(15, 23, 42, 0.35)',
                      transition: 'background-color 120ms ease, border 120ms ease',
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: `${color}22`, color }}>
                        <IconComponent fontSize="small" />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Typography fontWeight={600} display="flex" alignItems="center" gap={1}>
                          <Typography component="span" variant="caption" color="text.secondary">
                            {String(index + 1).padStart(2, '0')}
                          </Typography>
                          {step.title}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {label}
                          {step.description ? ' • Hover for details' : ''}
                        </Typography>
                      }
                    />
                  </ListItem>
                </Tooltip>
              );
            })}
          </List>
        )}
      </CardContent>
    </Card>
  );
};
