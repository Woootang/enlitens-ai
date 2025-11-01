import { Badge, Card, CardBody, CircularProgress, CircularProgressLabel, Divider, Stack, Text } from '@chakra-ui/react';
import { QualityMetric } from '../types';

interface QualityPanelProps {
  metrics: QualityMetric[];
  layerFailures: string[];
}

const getMetricColor = (metric: QualityMetric) => {
  const ratio = metric.value / metric.target;
  if (metric.direction === 'higher-is-better') {
    if (ratio >= 1) return 'green';
    if (ratio >= 0.75) return 'yellow';
    return 'red';
  }
  // lower is better
  if (metric.value <= metric.target) return 'green';
  if (metric.value <= metric.target * 1.25) return 'yellow';
  return 'red';
};

export const QualityPanel = ({ metrics, layerFailures }: QualityPanelProps) => (
  <Card borderRadius="xl" borderColor="whiteAlpha.200">
    <CardBody as={Stack} spacing={5}>
      <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
        Quality Metrics
      </Text>

      <Stack direction={{ base: 'column', md: 'row' }} spacing={6} align="stretch">
        {metrics.length === 0 ? (
          <Text fontSize="sm" color="slate.300">
            Waiting for validation telemetry.
          </Text>
        ) : (
          metrics.slice(0, 3).map((metric) => {
            const color = getMetricColor(metric);
            const progress = metric.direction === 'higher-is-better'
              ? Math.min(100, (metric.value / metric.target) * 100)
              : Math.min(100, (metric.target / Math.max(metric.value, 1e-6)) * 100);
            return (
              <Stack key={metric.id} align="center" spacing={3} flex={1}>
                <CircularProgress
                  value={Number.isFinite(progress) ? progress : 0}
                  size="140px"
                  thickness="12px"
                  color={`${color}.300`}
                  trackColor="whiteAlpha.100"
                >
                  <CircularProgressLabel>
                    <Text fontSize="lg" fontWeight="semibold" color="white">
                      {metric.value.toFixed(2)}
                    </Text>
                  </CircularProgressLabel>
                </CircularProgress>
                <Stack spacing={1} align="center">
                  <Text fontSize="sm" color="slate.100" textAlign="center">
                    {metric.label}
                  </Text>
                  <Badge variant="subtle" colorScheme={color}>
                    Target {metric.target.toFixed(2)} · {metric.direction.replace('-', ' ')}
                  </Badge>
                </Stack>
              </Stack>
            );
          })
        )}
      </Stack>

      <Divider borderColor="whiteAlpha.200" />

      <Stack spacing={2}>
        <Text fontSize="sm" color="slate.200">
          Validation Flags
        </Text>
        {layerFailures.length === 0 ? (
          <Text fontSize="sm" color="slate.400">
            No outstanding validation issues.
          </Text>
        ) : (
          layerFailures.map((failure, index) => (
            <Badge key={`${failure}-${index}`} colorScheme="red" variant="subtle" alignSelf="flex-start">
              {failure}
            </Badge>
          ))
        )}
      </Stack>
    </CardBody>
  </Card>
);
