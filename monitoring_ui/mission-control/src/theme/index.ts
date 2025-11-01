import { extendTheme, type ThemeConfig } from '@chakra-ui/react';

const config: ThemeConfig = {
  initialColorMode: 'dark',
  useSystemColorMode: false,
};

export const theme = extendTheme({
  config,
  fonts: {
    heading: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    body: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  styles: {
    global: {
      body: {
        bg: 'gray.900',
        color: 'slate.100',
      },
    },
  },
  colors: {
    brand: {
      50: '#e4f0ff',
      100: '#bed4ff',
      200: '#96b9ff',
      300: '#6e9efe',
      400: '#467ffd',
      500: '#2d65e4',
      600: '#1f4db2',
      700: '#153580',
      800: '#0b1f4f',
      900: '#050d27',
    },
  },
  components: {
    Card: {
      baseStyle: {
        container: {
          bg: 'rgba(15, 23, 42, 0.8)',
          borderWidth: '1px',
          borderColor: 'whiteAlpha.200',
          backdropFilter: 'blur(12px)',
        },
      },
    },
  },
});
