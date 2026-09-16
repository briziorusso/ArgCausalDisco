aspcr_is_dag <- function(G) {
  ## HEJ stores directed edges as G[child, parent].  Convert to the repository
  ## convention (row -> column) before applying Kahn's algorithm.
  A <- t(G != 0)
  diag(A) <- FALSE
  indegree <- colSums(A)
  queue <- which(indegree == 0)
  visited <- 0L
  while (length(queue) > 0) {
    node <- queue[[1]]
    queue <- queue[-1]
    visited <- visited + 1L
    children <- which(A[node, ])
    for (child in children) {
      indegree[[child]] <- indegree[[child]] - 1
      if (indegree[[child]] == 0) queue <- c(queue, child)
    }
  }
  visited == nrow(A)
}


aspcr_safe_ratio <- function(numerator, denominator) {
  if (denominator == 0) return(NA_real_)
  as.numeric(numerator) / as.numeric(denominator)
}


aspcr_constraint_audit <- function(global_indeps, truth_model, learned_model,
                                   weight = "log") {
  rows <- lapply(seq_along(global_indeps), function(index) {
    fact <- global_indeps[[index]]
    tested_independent <- isTRUE(fact$independent)
    truth_independent <- !directed_reachable(
      fact$vars[[1]], fact$vars[[2]], fact$C, fact$J, truth_model
    )
    final_independent <- !directed_reachable(
      fact$vars[[1]], fact$vars[[2]], fact$C, fact$J, learned_model
    )
    test_correct <- tested_independent == truth_independent
    retained <- tested_independent == final_independent
    raw_weight <- if (weight == "log") as.numeric(fact$w) else 1
    asp_weight <- if (weight == "log") round(1000 * raw_weight) else 1

    data.frame(
      constraint_id = as.integer(index),
      x = as.integer(fact$vars[[1]]),
      y = as.integer(fact$vars[[2]]),
      conditioning_set = paste(as.integer(fact$C), collapse = ";"),
      intervention_set = paste(as.integer(fact$J), collapse = ";"),
      cset = as.integer(fact$cset),
      jset = as.integer(fact$jset),
      mset = as.integer(fact$mset),
      test_independent = tested_independent,
      truth_independent = truth_independent,
      final_independent = final_independent,
      tested_relation = if (tested_independent) "independent" else "dependent",
      truth_relation = if (truth_independent) "independent" else "dependent",
      final_graph_relation = if (final_independent) "independent" else "dependent",
      test_correct = test_correct,
      retained = retained,
      retained_true = retained && test_correct,
      retained_false = retained && !test_correct,
      raw_weight = raw_weight,
      asp_weight = as.integer(asp_weight),
      probability_independent = as.numeric(fact$p),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}


aspcr_fact_diagnostics <- function(audit) {
  fact_total <- nrow(audit)
  fact_true <- sum(audit$test_correct)
  fact_false <- fact_total - fact_true
  fact_retained <- sum(audit$retained)
  fact_retained_true <- sum(audit$retained_true)
  fact_retained_false <- sum(audit$retained_false)
  fact_removed <- fact_total - fact_retained
  fact_removed_true <- fact_true - fact_retained_true
  fact_removed_false <- fact_false - fact_retained_false
  fact_precision <- aspcr_safe_ratio(fact_retained_true, fact_retained)
  fact_recall <- aspcr_safe_ratio(fact_retained_true, fact_true)
  fact_f1 <- if (is.na(fact_precision) || is.na(fact_recall) ||
                    fact_precision + fact_recall == 0) {
    0
  } else {
    2 * fact_precision * fact_recall / (fact_precision + fact_recall)
  }

  list(
    fact_total = fact_total,
    fact_true = fact_true,
    fact_false = fact_false,
    fact_retained = fact_retained,
    fact_retained_true = fact_retained_true,
    fact_retained_false = fact_retained_false,
    fact_removed = fact_removed,
    fact_removed_true = fact_removed_true,
    fact_removed_false = fact_removed_false,
    fact_precision = fact_precision,
    fact_recall = fact_recall,
    fact_f1 = fact_f1,
    recomputed_objective = sum(audit$asp_weight[!audit$retained])
  )
}


runpipeexternal <- function(dat_loc, algo = "log-weights", N_override = NULL,
                            model_space = "dag_sufficient") {
  ## Run ASPCR on an externally generated matched data set and retain enough
  ## evidence to audit every CI fact and the optimiser objective.
  exconf <- "passive"

  if (algo == "log-weights" || algo == 1) {
    test <- "bayes"
    weight <- "log"
    p <- 0.4
    alpha <- 20
    solver <- "clingo"
    if (model_space == "dag_sufficient") {
      encode <- "new_wmaxsat_acyclic_sufficient.pl"
    } else if (model_space == "general") {
      encode <- "new_wmaxsat.pl"
    } else {
      stop("Unknown ASPCR model_space: ", model_space)
    }
  } else if (algo == "hard-deps" || algo == 2) {
    test <- "classic"
    weight <- "constant"
    p <- 0.001
    alpha <- NA_real_
    solver <- "clingo"
    encode <- "new_maxindep.pl"
  } else if (algo == "constant-weights" || algo == 3) {
    test <- "classic"
    weight <- "constant"
    p <- 0.05
    alpha <- NA_real_
    solver <- "clingo"
    encode <- if (model_space == "dag_sufficient") {
      "new_wmaxsat_acyclic_sufficient.pl"
    } else {
      "new_wmaxsat.pl"
    }
  } else {
    stop("Unsupported algorithm for auditable ASPCR experiments: ", algo)
  }

  X <- as.matrix(read.csv(dat_loc, header = FALSE, sep = ",", check.names = FALSE))
  storage.mode(X) <- "numeric"
  N <- if (is.null(N_override)) nrow(X) else as.integer(N_override)
  if (N != nrow(X)) {
    warning("N_override does not match dat_loc rows; it is retained as metadata only.")
  }

  G_loc <- sub("data_", "true_graph_", dat_loc, fixed = TRUE)
  G <- as.matrix(read.csv(G_loc, header = FALSE, sep = ",", check.names = FALSE))
  storage.mode(G) <- "numeric"
  if (nrow(G) != ncol(G)) stop("The true graph must be square.")
  if (ncol(X) != nrow(G)) stop("Data columns do not match the true graph.")

  n <<- nrow(G)
  global_n <<- n
  schedule <<- n - 2
  global_indeps <<- list()

  ## HEJ uses G[child, parent]; Python writes B_true[parent, child].
  M <- list(G = t(G), Ge = array(0, c(n, n)), Gs = array(0, c(n, n)))
  M$B <- M$G * matrix(runif(n * n, 0.2, 0.8), n, n) *
    matrix(sample(c(-1, 1), n * n, replace = TRUE), n, n)
  M$Ce <- diag(abs(1 + 0.1 * rnorm(n)))
  D <- list(list(e = rep(0, n), M = M, data = X, N = nrow(X)))

  start_time <- Sys.time()
  L <- learn(
    D, test = test, schedule = schedule, weight = weight, encode = encode,
    p = p, alpha = alpha,
    clingoconf = "--configuration=crafty --time-limit=25000 --quiet=1,0",
    verbose = 1
  )
  elapsed_total <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  if (is.null(L$solving_time) || is.infinite(L$solving_time) || is.null(L$G)) {
    stop("ASPCR solver did not return a finite graph solution.")
  }

  L$Ge <- if (is.null(L$Ge)) array(0, c(n, n)) else L$Ge
  L$Gs <- if (is.null(L$Gs)) array(0, c(n, n)) else L$Gs
  matrices <- list(directed = L$G, bidirected = L$Ge, tailtail = L$Gs)
  for (matrix_name in names(matrices)) {
    component <- matrices[[matrix_name]]
    if (!identical(dim(component), c(n, n))) {
      stop("Invalid ", matrix_name, " graph dimensions.")
    }
    if (any(!component %in% c(0, 1))) stop("Non-binary ", matrix_name, " graph.")
    if (any(diag(component) != 0)) stop("Self-edge in ", matrix_name, " graph.")
  }

  graph_is_dag <- aspcr_is_dag(L$G)
  n_directed <- sum(L$G)
  n_bidirected <- sum(L$Ge) / 2
  n_tailtail <- sum(L$Gs) / 2
  if (model_space == "dag_sufficient") {
    if (!graph_is_dag) stop("DAG ASPCR encoding returned a directed cycle.")
    if (n_bidirected != 0) stop("DAG ASPCR encoding returned bidirected edges.")
    if (n_tailtail != 0) stop("DAG ASPCR encoding returned tail-tail edges.")
  }

  expected_constraints <- choose(n, 2) * 2^(n - 2)
  if (length(global_indeps) != expected_constraints) {
    stop("Expected ", expected_constraints, " CI constraints, got ", length(global_indeps), ".")
  }
  audit <- aspcr_constraint_audit(global_indeps, M, L, weight = weight)
  facts <- aspcr_fact_diagnostics(audit)
  solver_objective <- as.numeric(L$objective)
  recomputed_objective <- as.numeric(facts$recomputed_objective)
  if (!is.finite(solver_objective) || solver_objective != recomputed_objective) {
    stop("ASPCR objective mismatch: solver=", solver_objective,
         ", recomputed=", recomputed_objective, ".")
  }

  out_dir <- sub("data$", "results", dirname(dat_loc))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  run_id <- sub("\\.csv$", "", sub("^data_", "", basename(dat_loc)))
  prefix <- file.path(
    out_dir,
    paste0("aspcr_", gsub("[^A-Za-z0-9_.-]", "_", algo), "_",
           gsub("[^A-Za-z0-9_.-]", "_", model_space), "_", run_id)
  )
  paths <- list(
    directed = paste0(prefix, "_directed.csv"),
    bidirected = paste0(prefix, "_bidirected.csv"),
    tailtail = paste0(prefix, "_tailtail.csv"),
    constraints = paste0(prefix, "_constraints.csv"),
    diagnostics = paste0(prefix, "_diagnostics.csv"),
    time = paste0(prefix, "_time.csv")
  )

  diagnostics <- data.frame(
    algorithm = as.character(algo),
    model_space = model_space,
    encoding = encode,
    test = test,
    weight = weight,
    prior_independence = p,
    alpha = alpha,
    sample_size = N,
    n_nodes = n,
    expected_constraints = expected_constraints,
    fact_total = facts$fact_total,
    fact_true = facts$fact_true,
    fact_false = facts$fact_false,
    fact_retained = facts$fact_retained,
    fact_retained_true = facts$fact_retained_true,
    fact_retained_false = facts$fact_retained_false,
    fact_removed = facts$fact_removed,
    fact_removed_true = facts$fact_removed_true,
    fact_removed_false = facts$fact_removed_false,
    fact_precision = facts$fact_precision,
    fact_recall = facts$fact_recall,
    fact_f1 = facts$fact_f1,
    solver_objective = solver_objective,
    recomputed_objective = recomputed_objective,
    graph_is_dag = graph_is_dag,
    n_directed = n_directed,
    n_bidirected = n_bidirected,
    n_tailtail = n_tailtail,
    elapsed_total = elapsed_total,
    testing_time = as.numeric(L$testing_time),
    encoding_time = as.numeric(L$encoding_time),
    solving_time = as.numeric(L$solving_time),
    data_path = normalizePath(dat_loc),
    true_graph_path = normalizePath(G_loc),
    stringsAsFactors = FALSE
  )

  write.table(L$G, paths$directed, sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(L$Ge, paths$bidirected, sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(L$Gs, paths$tailtail, sep = ",", row.names = FALSE, col.names = FALSE)
  write.csv(audit, paths$constraints, row.names = FALSE)
  write.csv(diagnostics, paths$diagnostics, row.names = FALSE)
  write.table(elapsed_total, paths$time, sep = ",", row.names = FALSE, col.names = FALSE)

  invisible(list(graph = L, truth = M, constraints = audit,
                 diagnostics = diagnostics, paths = paths))
}
