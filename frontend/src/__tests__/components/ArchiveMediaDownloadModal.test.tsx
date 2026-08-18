import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ArchiveMediaDownloadModal } from '../../components/ArchiveMediaDownloadModal';
import { api } from '../../api/client';

const showToast = vi.fn();

vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => ({ showToast }),
}));

vi.mock('../../api/client', () => ({
  api: {
    getArchivePrinterMedia: vi.fn(),
    getArchiveTimelapse: vi.fn(() => '/api/v1/archives/1/timelapse'),
    downloadPrinterFilesAsZip: vi.fn(),
  },
}));

const media = {
  archive_id: 1,
  printer_id: 1,
  local_timelapse: null,
  remote_files: [{
    name: 'video.mp4',
    path: '/timelapse/video.mp4',
    size: 1024,
    mtime: '2026-08-18T10:00:00Z',
    kind: 'timelapse' as const,
  }],
  warnings: [],
};

describe('ArchiveMediaDownloadModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getArchivePrinterMedia).mockResolvedValue(media);
  });

  it('does not reselect a manually deselected single file when query data refreshes', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={queryClient}>
        <ArchiveMediaDownloadModal
          archiveId={1}
          archiveName="Test print"
          printerName="Printer"
          onClose={vi.fn()}
        />
      </QueryClientProvider>,
    );

    const filename = await screen.findByText('video.mp4');
    const downloadButton = screen.getByRole('button', { name: /Download selected \(1\)/i });
    expect(downloadButton).toBeEnabled();

    fireEvent.click(filename.closest('button')!);
    expect(screen.getByRole('button', { name: /Download selected \(0\)/i })).toBeDisabled();

    await act(async () => {
      queryClient.setQueryData(['archive-printer-media', 1], {
        ...media,
        remote_files: [...media.remote_files],
      });
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Download selected \(0\)/i })).toBeDisabled();
    });
  });
});
