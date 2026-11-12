//! Test support: re-run the current test binary on one ignored test in a child process, check
//! that it really ran, and check that the thread pool runs tasks on its workers.

use std::process::{Command, Output};

/// Runs the ignored test `test` of this test binary in a child process, alone, with the
/// variables in `env` set, and returns what the child wrote. Alone in its process, nothing
/// else holds the thread pool, so a call on several threads is never sent to the inline path
/// because another test's call is using the pool. The child inherits every other variable.
pub(crate) fn run_alone(test: &str, env: &[(&str, &str)]) -> Output {
    run_alone_without(test, env, &[])
}

/// Like [`run_alone`], but the variables named in `remove` are removed from the child's
/// environment first.
pub(crate) fn run_alone_without(test: &str, env: &[(&str, &str)], remove: &[&str]) -> Output {
    let exe = std::env::current_exe().expect("the test binary's path");
    let mut command = Command::new(exe);
    command.args([
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
        test,
    ]);
    for name in remove {
        command.env_remove(name);
    }
    for (name, value) in env {
        command.env(name, value);
    }
    command.output().expect("the test binary runs")
}

/// Panics unless the child `out` exited successfully and its summary says one test passed. A
/// filter that matches no test also exits 0, so the summary shows the test ran.
pub(crate) fn assert_passed(out: &Output) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "the child should run the test and pass; stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Fails unless the thread pool runs tasks on its workers, not only on the caller: runs 64
/// short tasks on 4 threads, again for up to 10 seconds, until one runs on a worker other
/// than 0.
pub(crate) fn assert_pool_runs_on_workers() {
    use cavitation_core::kernels::pool::{Task, run};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let start = std::time::Instant::now();
    loop {
        let highest = AtomicUsize::new(0);
        run(4, 64, &|t: Task| {
            let spin = std::time::Instant::now();
            while spin.elapsed() < std::time::Duration::from_micros(20) {}
            highest.fetch_max(t.worker, Ordering::Relaxed);
        });
        if highest.into_inner() > 0 {
            return;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "the pool ran every task on the caller for 10 s"
        );
    }
}
