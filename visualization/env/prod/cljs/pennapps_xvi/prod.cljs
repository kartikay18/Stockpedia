(ns pennapps-xvi.prod
  (:require
    [pennapps-xvi.core :as core]))

;;ignore println statements in prod
(set! *print-fn* (fn [& _]))

(core/init!)
