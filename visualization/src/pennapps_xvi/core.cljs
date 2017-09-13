(ns pennapps-xvi.core
    (:require
     [clojure.core]
     [reagent.core :as r]
     [ajax.core :refer [GET POST]]))

(defn floor-factor [n f]
  (* f (quot n f)))

(defn gen-dummy-stock-data
  [len]
  (let [rand-range 0.8
        rand-range-half (/ rand-range 2)]
    (take
     len
     (iterate (fn [{:strs [open high low close volume date]}]
                {"open" close
                 "high" (* close (+ 1.0 (* (- (rand rand-range) rand-range-half) (- (rand rand-range) rand-range-half))))
                 "low" (* close (+ 1.0 (* (- (rand rand-range) rand-range-half) (- (rand rand-range) rand-range-half))))
                 "close" (* close (+ 1.001 (* (- (rand rand-range) rand-range-half) (- (rand rand-range) rand-range-half))))
                 "volume" (* volume (+ 1.0 (* (- (rand rand-range) rand-range-half) (- (rand rand-range) rand-range-half) (- (rand rand-range) rand-range-half))))
                 "date" (js/Date. (+ (.getTime date) 86400000))})
              {"date" (js/Date. (floor-factor (js/Date.now) 86400000))
               "open" 62.40
               "high" 63.34
               "low" 61.79
               "close" 62.88
               "volume" 37617413}))))


;; -------------------------
;; Views

(def dummy-portfolio-data
  [
   ["U.S. Stocks" "1123"]
   ["U.S. Bonds" "2287"]
   ["Cryptocurrencies" "3419"]
   ["Energy" "4200"]
   ["International Stock" "877"]
   ["Emerging Markets" "1877"]
   ])

(def friend-list
  [
   ["1" "Jennifer" "AAPL"]
   ["2" "Jake" "GOOG"]
   ["3" "Mike" "AMZN"]
   ["4" "Sam" "MSFT"]
   ["5" "Tim" "VTSMX"]
   ["6" "Julie" "F"]
   ["7" "Harry" "SYMC"]
   ["8" "Jolene" "XOM"]
   ["9" "William" "BLK"]
   ["10" "Elaina" "BLK"]
   ["11" "Shu" "CSCO"]
   ["12" "Irene" "FDX"]
   ["13" "Kelly" "FB"]])

(def chart-colors
  ["#A96186" "#A96162" "#A98461" "#A9A861" "#86A961" "#589857" "#4E876A" "#4E8786" "#4E6B87" "#4E4E87" "#6A4E87" "#8F5390"])

(defn overview []
  (r/create-class
   {:component-did-mount
    (fn [this]
      (set! js/window.d3 js/window.d3v3)
      (js/c3.generate (clj->js {"bindto" "#portfolio-composition"
                                "color" {"pattern"
                                         (clojure.core/shuffle chart-colors)}
                                "data" {"columns" dummy-portfolio-data
                                        "type" "donut"}}))
      )
    :reagent-render
    (fn [_] [:div
             [:h2 "Overview"]
             [:h3 "Your Investment Portfolio"]
             [:div#portfolio-composition]
             [:h3 "Friends\u2019 Picks"]
             [:div#stock-picks
              (map (fn [[id name pick]]
                     [:div.stock-pick {:key name}
                      [:div [:img {:src (str "img/profile/pic-" id ".jpg")}]]
                      [:div.name name]
                      [:div.pick pick]])
                   (take 5 (clojure.core/shuffle friend-list)))]
             ])}))

(defn investments []
  [:h2 "Long-Term Investments"])

(defonce current-search (r/atom "AAPL"))

(defonce current-range (r/atom 60))

(defonce current-data (r/atom (gen-dummy-stock-data 60)))

(defn candlestick [raw-data range]
  (let [r (js/Math.floor (* 1000000000 (rand)))
        dorender
        (fn [this]
        (set! js/window.d3 js/window.d3v4)
        (let [margin-top 20
              margin-right 20
              margin-bottom 30
              margin-left 50
              width (- 520 margin-left margin-right)
              height (- 300 margin-top margin-bottom)
              x (-> js/techan.scale (.financetime) (.range (clj->js [0 width])))
              y (-> js/d3 (.scaleLinear) (.range (clj->js [height 0])))
              candlestick (-> js/techan.plot (.candlestick) (.xScale x) (.yScale y))
              xAxis (-> js/d3 (.axisBottom) (.scale x))
              yAxis (-> js/d3 (.axisLeft) (.scale y))
              svg (-> js/d3
                      (.select (str "div.candlestick.id" r))
                      (.html "")
                      (.append "svg")
                      (.attr "class" "candlestick")
                      (.attr "width" (+ width margin-left margin-right))
                      (.attr "height" (+ height margin-top margin-bottom))
                      (.append "g")
                      (.attr "transform" (str "translate(" margin-left "," margin-top ")")))
              accessor (-> candlestick (.accessor))
              data (-> (clj->js (take @range @raw-data)) (.sort (fn [a b] (js/d3.ascending (.d accessor a) (.d accessor b)))))
              ]
          (-> svg (.append "g") (.attr "class" "candlestick"))
          (-> svg (.append "g") (.attr "class" "x axis") (.attr "transform" (str "translate(0," height ")")))
          (-> svg (.append "g") (.attr "class" "y axis") (.append "text") (.attr "transform" "rotate(-90)") (.attr "y" "6")
              (.attr "dy" ".71em") (.style "text-anchor" "end") (.text "Price ($)"))
          (.domain x (.map data (.-d (-> candlestick (.accessor)))))
          (.domain y (.domain (js/techan.scale.plot.ohlc data (-> candlestick (.accessor)))))
          (-> svg (.selectAll "g.candlestick") (.datum data) (.call candlestick))
          (-> svg (.selectAll "g.x.axis") (.call xAxis))
          (-> svg (.selectAll "g.y.axis") (.call yAxis))
          ))]
    (r/create-class
     {:component-did-mount dorender
      :component-did-update dorender
      :reagent-render
      (fn [_]
        [:div {:display "none"} (str @raw-data @range)]
        [:div.candlestick {:class (str "id" r)}])})))

(defn trading []
  [:div
   [:h2 "Short-Term Trading"]
   [:input {:type "text" :placeholder "Search for a stock symbol..." :class "ticksymbol"
            :value @current-search :on-change #(swap! current-search (constantly (-> % .-target .-value)))}]
   [:div
    [:span "Days to Predict (max 60)"]
    [:input {:type "range" :min "1" :max "60" :step "1"
             :on-change #(swap! current-range (constantly (js/window.parseInt (-> % .-target .-value) 10)))
}]]
   [:button {:on-click #(do
                          (swap! current-data (constantly (gen-dummy-stock-data 60)))
                          (POST "https://data.chastiser11.hasura-app.io/v1/query"
                              {:headers
                               {
                                "authorization" "Bearer wj7fmf21w6lvu0l4u7vmdef1tqo0cykn"}
                               :format :json
                               :response-format :json
                               :params
                               {"type" "select", "args" {"table"  "stockTest", "columns"  ["*"], "where" { "stocksymbol"  @current-search }}}
                               :handler
                               (fn [response]
                                 (let [today (js/Date. (floor-factor (js/Date.now) 86400000))
                                       predictions (js->clj (js/window.JSON.parse (get-in response [0 "predictions"])))
                                       predictions-t (apply mapv vector predictions)
                                       transformed (map-indexed (fn [i [o h l c]]
                                                                  {"open" o
                                                                   "high" h
                                                                   "low" l
                                                                   "close" c
                                                                   "date" (js/Date. (+ (.getTime today) (* i 86400000)))
                                                                   }
                                                                  ) (take 60 predictions-t))]
                                   (if-not (empty? transformed)
                                     (swap! current-data (constantly transformed)))))}))} "Load Prediction"]
   [candlestick current-data current-range]])

(defn analytics []
  [:h2 "Analytics"])

(def tabmap
  {"Overview" overview
   "Long-Term Investments" investments
   "Short-Term Trading" trading
   "Analytics" analytics})

(def tabs
  (keys tabmap))

(defonce current-tab (r/atom (or (-> js/window.localStorage (.getItem "current-tab")) "Overview")))

(defn body []
  [:div {:id "body"}
   [:div#sidebar
    [:ul
     (map (fn [t] [:li {:key t} [:a {:on-click #(do
                                                  (-> js/window.localStorage (.setItem "current-tab" t))
                                                  (swap! current-tab (fn [_] t)))
                                     :class (if (= @current-tab t) "selected" "")} t]]) tabs)]]
   [:div#main-content
    [(get tabmap @current-tab "Overview")]]])

(defn home-page []
      [:div#container
     [:div#header
      [:img#logo {:src "img/logo.png" :height 64 :width 65}]
      [:p#title "Stockpedia"]
      [:p#menu "Welcome Johnson! "
       [:button "Logout"]]]
     [body]
     [:div#footer
      [:p [:strong "Copyright (c) 2017 Stock Advisors and Company. All rights reserved."]]
      [:p [:strong "Important Disclaimer: "] "Any past performance, projection, forecast or simulation of results is not necessarily indicative of the future or likely performance of any company or investment. The information provided does not take into account your specific investment objectives, financial situation or particular needs. Before you act on any information on this site, you should always seek independent financial, tax, and legal advice or make such independent investigations as you consider necessary or appropriate regarding the suitability of the investment product, taking into account your specific investment objectives, financial situation or particular needs."]]]
  )

;; -------------------------
;; Initialize app

(defn mount-root []
  (r/render [home-page] (.getElementById js/document "app")))

(defn init! []
  (mount-root))
