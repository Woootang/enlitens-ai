import { Card, CardBody, Progress, Stack, Tag, Text, Wrap, WrapItem } from '@chakra-ui/react';
import { QualityMetric } from '../types';

interface QualityPanelProps {
  metrics: QualityMetric[];
  layerFailures: string[];
}

const severityPalette: Record<QualityMetric['severity'], { bar: string; background: string; text: string }> = {
  normal: { bar: '#34d399', background: 'rgba(52,211,153,0.12)', text: '#a7f3d0' },
  warning: { bar: '#facc15', background: 'rgba(250,204,21,0.15)', text: '#fef3c7' },
  critical: { bar: '#f87171', background: 'rgba(248,113,113,0.18)', text: '#fee2e2' },
};

const formatBaseline = (metric: QualityMetric) => {
  if (metric.baseline === null || metric.baseline === undefined || Number.isNaN(metric.baseline)) {
    return null;
  }
  const value = metric.baseline;
  if (value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(2);
};

const normalizeValue = (metric: QualityMetric) => {
  if (metric.value === null || metric.value === undefined) {
    return 0;
  }
  if (metric.orientation === 'lower_is_better') {
    const value = Math.min(Math.max(metric.value * 100, 0), 100);
    return 100 - value;
  }
  const numeric = metric.value <= 1 ? metric.value * 100 : Math.min(metric.value, 100);
  return Math.max(Math.min(numeric, 100), 0);
};

export const QualityPanel = ({ metrics, layerFailures }: QualityPanelProps) => {
  return (
    <Card borderRadius="xl" borderColor="whiteAlpha.200">
      <CardBody as={Stack} spacing={5}>
        <Stack spacing={0.5}>
          <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
            Quality Metrics
          </Text>
          <Text fontSize="sm" color="slate.300">
            Faithfulness, alignment, and citation integrity checks
          </Text>
        </Stack>

        <Stack spacing={3}>
          {metrics.length === 0 && (
            <Text fontSize="sm" color="slate.300">
              Awaiting quality telemetry…
            </Text>
          )}
          {metrics.map((metric) => {
            const palette = severityPalette[metric.severity];
            const baseline = formatBaseline(metric);
            return (
              <Stack key={metric.id} spacing={2}>
                <Stack direction="row" justify="space-between" align="flex-start">
                  <Stack spacing={0}>
                    <Text fontSize="sm" fontWeight="semibold">
                      {metric.label}
                    </Text>
                    <Text fontSize="xs" color="slate.400">
                      {metric.orientation === 'lower_is_better' ? 'Lower is better' : 'Higher is better'}
                    </Text>
                  </Stack>
                  <Stack spacing={0} align="flex-end">
                    <Text fontSize="lg" fontWeight="bold" color={palette.text}>
                      {metric.displayValue}
                    </Text>
                    {baseline && (
                      <Text fontSize="xs" color="slate.400">
                        Baseline {metric.orientation === 'lower_is_better' ? '≤' : '≥'} {baseline}
                      </Text>
                    )}
                  </Stack>
                </Stack>
                <Progress
                  value={normalizeValue(metric)}
                  height="10px"
                  borderRadius="full"
                  bg={palette.background}
                  sx={{ '& > div': { backgroundColor: palette.bar } }}
                />
              </Stack>
            );
          })}
        </Stack>

        {layerFailures.length > 0 && (
          <Stack spacing={2}>
            <Text fontSize="sm" color="slate.300">
              Validation findings
            </Text>
            <Wrap spacing={2}>
              {layerFailures.map((failure, index) => (
                <WrapItem key={`${failure}-${index}`}>
                  <Tag colorScheme="yellow" variant="outline" borderRadius="full" px={3} py={1} fontSize="xs">
                    {failure}
                  </Tag>
                </WrapItem>
              ))}
            </Wrap>
          </Stack>
        )}
      </CardBody>
    </Card>
  );
};
