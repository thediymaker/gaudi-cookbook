# SPDX-License-Identifier: Apache-2.0
"""Block SIGPIPE for this process and every thread and child it starts.

Import this first, before torch or habana_frameworks load and before any thread exists. The Habana runtime
installs a native SIGPIPE handler over Python's SIG_IGN and treats the signal as a device fatality ("Synapse is
being terminated due to fatality", synStatus 31). One write to a closed socket or pipe (a listener that hung up
mid-stream, an encoder subprocess that exited early) would end the HPU context and every request after it. A
blocked signal never reaches that handler: the write fails with EPIPE and Python raises BrokenPipeError as usual.
Threads and child processes inherit the mask, which is why it must be set in the main thread at startup.
"""

import signal

signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGPIPE})


def is_blocked() -> bool:
    return signal.SIGPIPE in signal.pthread_sigmask(signal.SIG_BLOCK, set())
