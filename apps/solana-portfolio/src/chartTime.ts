import type {Candle,Interval} from "./types";
const steps = {"1h":3600,"4h":14400,"1d":86400,"1w":604800};
export function candleStart(time:number,interval:Interval) {
  if(interval === "1M") {const d=new Date(time*1000);return Date.UTC(d.getUTCFullYear(),d.getUTCMonth(),1)/1000;}
  const step=steps[interval];
  const anchor=interval === "1w" ? 345600 : 0;
  return Math.floor((time-anchor)/step)*step+anchor;
}
export function candleEnd(candle:Candle,interval:Interval) {
  if(candle.endTime != null) return candle.endTime;
  if(interval === "1M") {const d=new Date(candle.time*1000);return Date.UTC(d.getUTCFullYear(),d.getUTCMonth()+1,1)/1000;}
  return candle.time+steps[interval];
}
