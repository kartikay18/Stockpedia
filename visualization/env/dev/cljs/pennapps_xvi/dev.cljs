(ns ^:figwheel-no-load pennapps-xvi.dev
  (:require
    [pennapps-xvi.core :as core]
    [devtools.core :as devtools]))


(enable-console-print!)

(devtools/install!)

(core/init!)
