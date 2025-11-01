import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#63b3ed',
    },
    background: {
      default: '#0f172a',
      paper: 'rgba(17, 25, 40, 0.8)',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5f5',
    },
  },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backdropFilter: 'blur(10px)',
        },
      },
    },
  },
});
